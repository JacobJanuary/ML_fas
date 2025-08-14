-- Создание таблицы для хранения ML предсказаний
CREATE SCHEMA IF NOT EXISTS ml;

-- Таблица предсказаний
CREATE TABLE IF NOT EXISTS ml.signal_predictions (
    id SERIAL PRIMARY KEY,

    -- Связь с исходным сигналом
    signal_id INTEGER NOT NULL REFERENCES fas.scoring_history(id),
    signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    trading_pair_id INTEGER NOT NULL,
    pair_symbol VARCHAR(20) NOT NULL,

    -- Тип и характеристики сигнала
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
    total_score NUMERIC NOT NULL,
    market_regime VARCHAR(20),

    -- ML предсказание
    model_version VARCHAR(50) NOT NULL, -- версия модели (например, 'adaptive_7d_v1')
    prediction_probability NUMERIC(5,4) NOT NULL CHECK (prediction_probability >= 0 AND prediction_probability <= 1),
    threshold_used NUMERIC(5,4) NOT NULL,
    prediction BOOLEAN NOT NULL, -- true = торговать, false = пропустить
    confidence_level VARCHAR(10) CHECK (confidence_level IN ('HIGH', 'MEDIUM', 'LOW')),

    -- Временные метки
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Дополнительная информация
    features_hash VARCHAR(64), -- хэш использованных фич для отладки
    model_metadata JSONB, -- дополнительная информация о модели

    -- Индекс для быстрого поиска
    CONSTRAINT unique_signal_prediction UNIQUE (signal_id)
);

-- Индексы для быстрых запросов
CREATE INDEX idx_predictions_signal_id ON ml.signal_predictions(signal_id);
CREATE INDEX idx_predictions_pair_id ON ml.signal_predictions(trading_pair_id);
CREATE INDEX idx_predictions_timestamp ON ml.signal_predictions(signal_timestamp DESC);
CREATE INDEX idx_predictions_predicted_at ON ml.signal_predictions(predicted_at DESC);
CREATE INDEX idx_predictions_signal_type ON ml.signal_predictions(signal_type);
CREATE INDEX idx_predictions_prediction ON ml.signal_predictions(prediction) WHERE prediction = true;

-- Таблица для отслеживания результатов предсказаний (для анализа эффективности)
CREATE TABLE IF NOT EXISTS ml.prediction_outcomes (
    id SERIAL PRIMARY KEY,

    -- Связь с предсказанием
    prediction_id INTEGER NOT NULL REFERENCES ml.signal_predictions(id),
    signal_id INTEGER NOT NULL,

    -- Фактический результат
    actual_outcome BOOLEAN, -- true = win, false = loss
    outcome_determined_at TIMESTAMP WITH TIME ZONE,
    max_favorable_move NUMERIC,
    max_adverse_move NUMERIC,
    time_to_outcome_hours NUMERIC,

    -- Метрики точности
    prediction_correct BOOLEAN GENERATED ALWAYS AS (
        CASE
            WHEN actual_outcome IS NOT NULL THEN
                (prediction_id IN (SELECT id FROM ml.signal_predictions WHERE prediction = true) AND actual_outcome = true)
                OR
                (prediction_id IN (SELECT id FROM ml.signal_predictions WHERE prediction = false) AND actual_outcome = false)
            ELSE NULL
        END
    ) STORED,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_outcomes_prediction_id ON ml.prediction_outcomes(prediction_id);
CREATE INDEX idx_outcomes_signal_id ON ml.prediction_outcomes(signal_id);
CREATE INDEX idx_outcomes_determined_at ON ml.prediction_outcomes(outcome_determined_at);

-- Представление для анализа эффективности моделей
CREATE OR REPLACE VIEW ml.model_performance AS
SELECT
    sp.model_version,
    sp.signal_type,
    DATE(sp.predicted_at) as prediction_date,
    COUNT(*) as total_predictions,
    COUNT(po.actual_outcome) as completed_outcomes,

    -- Метрики для предсказаний "торговать"
    COUNT(*) FILTER (WHERE sp.prediction = true) as signals_to_trade,
    COUNT(*) FILTER (WHERE sp.prediction = true AND po.actual_outcome = true) as true_positives,
    COUNT(*) FILTER (WHERE sp.prediction = true AND po.actual_outcome = false) as false_positives,

    -- Win rate
    CASE
        WHEN COUNT(*) FILTER (WHERE sp.prediction = true AND po.actual_outcome IS NOT NULL) > 0
        THEN COUNT(*) FILTER (WHERE sp.prediction = true AND po.actual_outcome = true)::numeric /
             COUNT(*) FILTER (WHERE sp.prediction = true AND po.actual_outcome IS NOT NULL)
        ELSE NULL
    END as win_rate,

    -- Средняя вероятность для положительных предсказаний
    AVG(sp.prediction_probability) FILTER (WHERE sp.prediction = true) as avg_probability_for_trades

FROM ml.signal_predictions sp
LEFT JOIN ml.prediction_outcomes po ON sp.id = po.prediction_id
GROUP BY sp.model_version, sp.signal_type, DATE(sp.predicted_at)
ORDER BY prediction_date DESC, sp.signal_type;

-- Функция для обновления результатов предсказаний
CREATE OR REPLACE FUNCTION ml.update_prediction_outcomes()
RETURNS void AS $$
BEGIN
    -- Обновляем результаты для предсказаний старше 48 часов
    INSERT INTO ml.prediction_outcomes (prediction_id, signal_id, actual_outcome, outcome_determined_at)
    SELECT
        sp.id,
        sp.signal_id,
        mtd.target,
        NOW()
    FROM ml.signal_predictions sp
    INNER JOIN fas.ml_training_data_v2 mtd ON sp.signal_id = mtd.id
    WHERE sp.predicted_at < NOW() - INTERVAL '48 hours'
        AND NOT EXISTS (
            SELECT 1 FROM ml.prediction_outcomes po
            WHERE po.prediction_id = sp.id
        )
        AND mtd.target IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Создаем задание для регулярного обновления результатов (если используется pg_cron)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('update-prediction-outcomes', '0 */6 * * *', 'SELECT ml.update_prediction_outcomes();');