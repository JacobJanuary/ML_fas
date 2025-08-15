-- Multi-Stage Filtering System Database Schema
-- =============================================

-- 1. Таблица для хранения множественных моделей
CREATE TABLE IF NOT EXISTS ml.model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'quick_filter', 'regime_specific', 'precision'
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL')),
    market_regime VARCHAR(20), -- NULL for universal models, or 'BULL', 'BEAR', 'NEUTRAL'

    -- Параметры модели
    window_days INTEGER,
    min_train_samples INTEGER,
    threshold NUMERIC(5,4),
    features_used TEXT[], -- список используемых фич

    -- Метрики производительности
    train_win_rate NUMERIC(5,4),
    validation_win_rate NUMERIC(5,4),
    live_win_rate NUMERIC(5,4), -- обновляется в реальном времени
    total_predictions INTEGER DEFAULT 0,
    successful_predictions INTEGER DEFAULT 0,

    -- Метаданные
    model_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_trained_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    training_metrics JSONB,

    -- Версионирование
    version INTEGER DEFAULT 1,
    parent_model_id INTEGER REFERENCES ml.model_registry(id)
);

-- 2. Таблица для multi-stage предсказаний
CREATE TABLE IF NOT EXISTS ml.multistage_predictions (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER NOT NULL REFERENCES fas.scoring_history(id),
    signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    trading_pair_id INTEGER NOT NULL,
    pair_symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,

    -- Stage 1: Quick Filter
    stage1_model_id INTEGER REFERENCES ml.model_registry(id),
    stage1_probability NUMERIC(5,4),
    stage1_passed BOOLEAN,
    stage1_time_ms INTEGER,

    -- Stage 2: Regime Detection
    detected_regime VARCHAR(20),
    regime_confidence NUMERIC(5,4),

    -- Stage 3: Specialized Model
    stage3_model_id INTEGER REFERENCES ml.model_registry(id),
    stage3_probability NUMERIC(5,4),
    stage3_threshold NUMERIC(5,4),
    stage3_passed BOOLEAN,
    stage3_time_ms INTEGER,

    -- Stage 4: Precision Filter
    stage4_model_id INTEGER REFERENCES ml.model_registry(id),
    stage4_probability NUMERIC(5,4),
    stage4_dynamic_threshold NUMERIC(5,4), -- динамически откалиброванный
    stage4_passed BOOLEAN,
    stage4_time_ms INTEGER,

    -- Финальное решение
    final_decision BOOLEAN NOT NULL,
    final_confidence NUMERIC(5,4),
    total_processing_time_ms INTEGER,

    -- Результат (заполняется позже)
    actual_outcome BOOLEAN,
    outcome_determined_at TIMESTAMP WITH TIME ZONE,

    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_multistage_signal UNIQUE (signal_id)
);

-- 3. Таблица для калибровки threshold
CREATE TABLE IF NOT EXISTS ml.threshold_calibration (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml.model_registry(id),
    calibration_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Данные калибровки
    data_start_time TIMESTAMP WITH TIME ZONE,
    data_end_time TIMESTAMP WITH TIME ZONE,
    samples_used INTEGER,

    -- Калибровочная кривая (probability -> actual win rate)
    calibration_curve JSONB, -- [{prob: 0.5, win_rate: 0.6}, ...]

    -- Рекомендованные thresholds для разных целевых win rates
    threshold_for_70_wr NUMERIC(5,4),
    threshold_for_75_wr NUMERIC(5,4),
    threshold_for_80_wr NUMERIC(5,4),
    threshold_for_85_wr NUMERIC(5,4),
    threshold_for_90_wr NUMERIC(5,4),

    -- Выбранный threshold
    selected_threshold NUMERIC(5,4),
    target_win_rate NUMERIC(5,4),
    expected_trade_rate NUMERIC(5,4), -- процент сигналов, которые пройдут

    is_active BOOLEAN DEFAULT true
);

-- 4. Таблица для мониторинга производительности в реальном времени
CREATE TABLE IF NOT EXISTS ml.performance_monitor (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml.model_registry(id),
    monitoring_window VARCHAR(20), -- '1h', '6h', '24h', '7d'

    -- Метрики
    total_signals INTEGER,
    signals_passed INTEGER,
    true_positives INTEGER,
    false_positives INTEGER,

    win_rate NUMERIC(5,4),
    pass_rate NUMERIC(5,4), -- процент пропущенных сигналов

    avg_probability NUMERIC(5,4),
    min_probability NUMERIC(5,4),
    max_probability NUMERIC(5,4),

    -- Сравнение с целевыми показателями
    target_win_rate NUMERIC(5,4),
    win_rate_deviation NUMERIC(5,4), -- отклонение от цели

    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(model_id, monitoring_window, calculated_at)
);

-- 5. Индексы для производительности
CREATE INDEX idx_model_registry_active ON ml.model_registry(is_active, model_type, signal_type);
CREATE INDEX idx_multistage_predictions_signal ON ml.multistage_predictions(signal_id);
CREATE INDEX idx_multistage_predictions_time ON ml.multistage_predictions(predicted_at DESC);
CREATE INDEX idx_multistage_predictions_decision ON ml.multistage_predictions(final_decision, predicted_at DESC);
CREATE INDEX idx_threshold_calibration_active ON ml.threshold_calibration(model_id, is_active);
CREATE INDEX idx_performance_monitor_recent ON ml.performance_monitor(calculated_at DESC, model_id);

-- 6. Функция для автоматической калибровки threshold
CREATE OR REPLACE FUNCTION ml.calibrate_threshold(
    p_model_id INTEGER,
    p_target_win_rate NUMERIC DEFAULT 0.80,
    p_lookback_hours INTEGER DEFAULT 48
)
RETURNS NUMERIC AS $$
DECLARE
    v_optimal_threshold NUMERIC;
    v_calibration_data JSONB;
BEGIN
    -- Собираем данные для калибровки
    WITH probability_buckets AS (
        SELECT
            WIDTH_BUCKET(stage4_probability, 0, 1, 20) as bucket,
            AVG(stage4_probability) as avg_prob,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE actual_outcome = true) as wins,
            CASE
                WHEN COUNT(*) > 0
                THEN COUNT(*) FILTER (WHERE actual_outcome = true)::NUMERIC / COUNT(*)
                ELSE 0
            END as win_rate
        FROM ml.multistage_predictions
        WHERE stage4_model_id = p_model_id
            AND actual_outcome IS NOT NULL
            AND predicted_at >= NOW() - INTERVAL '1 hour' * p_lookback_hours
        GROUP BY bucket
        HAVING COUNT(*) >= 5 -- минимум 5 примеров в bucket
    ),
    calibration_curve AS (
        SELECT
            jsonb_agg(
                jsonb_build_object(
                    'probability', avg_prob,
                    'win_rate', win_rate,
                    'samples', total
                ) ORDER BY avg_prob
            ) as curve
        FROM probability_buckets
    )
    SELECT
        -- Находим минимальную вероятность, дающую целевой win rate
        MIN(CASE
            WHEN (obj->>'win_rate')::NUMERIC >= p_target_win_rate
            THEN (obj->>'probability')::NUMERIC
            ELSE 1.0
        END),
        curve
    INTO v_optimal_threshold, v_calibration_data
    FROM calibration_curve, jsonb_array_elements(curve) obj;

    -- Если не нашли подходящий threshold, используем консервативный
    IF v_optimal_threshold IS NULL OR v_optimal_threshold >= 1.0 THEN
        v_optimal_threshold := 0.85; -- консервативный default
    END IF;

    -- Сохраняем калибровку
    INSERT INTO ml.threshold_calibration (
        model_id,
        data_start_time,
        data_end_time,
        samples_used,
        calibration_curve,
        selected_threshold,
        target_win_rate
    )
    SELECT
        p_model_id,
        MIN(predicted_at),
        MAX(predicted_at),
        COUNT(*),
        v_calibration_data,
        v_optimal_threshold,
        p_target_win_rate
    FROM ml.multistage_predictions
    WHERE stage4_model_id = p_model_id
        AND predicted_at >= NOW() - INTERVAL '1 hour' * p_lookback_hours;

    -- Деактивируем старые калибровки
    UPDATE ml.threshold_calibration
    SET is_active = false
    WHERE model_id = p_model_id
        AND calibration_timestamp < NOW() - INTERVAL '5 minutes';

    RETURN v_optimal_threshold;
END;
$$ LANGUAGE plpgsql;

-- 7. Функция для выбора лучшей модели на основе производительности
CREATE OR REPLACE FUNCTION ml.select_best_model(
    p_signal_type VARCHAR,
    p_market_regime VARCHAR,
    p_model_type VARCHAR
)
RETURNS INTEGER AS $$
DECLARE
    v_best_model_id INTEGER;
BEGIN
    -- Выбираем модель с лучшим win rate за последние 24 часа
    SELECT model_id INTO v_best_model_id
    FROM ml.performance_monitor pm
    INNER JOIN ml.model_registry mr ON pm.model_id = mr.id
    WHERE mr.signal_type = p_signal_type
        AND mr.model_type = p_model_type
        AND (mr.market_regime = p_market_regime OR mr.market_regime IS NULL)
        AND mr.is_active = true
        AND pm.monitoring_window = '24h'
        AND pm.calculated_at >= NOW() - INTERVAL '1 hour'
    ORDER BY
        pm.win_rate DESC NULLS LAST,
        pm.total_signals DESC
    LIMIT 1;

    -- Если не нашли с метриками, берем последнюю активную
    IF v_best_model_id IS NULL THEN
        SELECT id INTO v_best_model_id
        FROM ml.model_registry
        WHERE signal_type = p_signal_type
            AND model_type = p_model_type
            AND (market_regime = p_market_regime OR market_regime IS NULL)
            AND is_active = true
        ORDER BY last_trained_at DESC NULLS LAST
        LIMIT 1;
    END IF;

    RETURN v_best_model_id;
END;
$$ LANGUAGE plpgsql;

-- 8. Представление для мониторинга multi-stage системы
CREATE OR REPLACE VIEW ml.multistage_performance AS
SELECT
    DATE(predicted_at) as prediction_date,
    signal_type,
    COUNT(*) as total_signals,

    -- Stage 1 metrics
    COUNT(*) FILTER (WHERE stage1_passed = true) as stage1_passed,
    AVG(stage1_time_ms) as avg_stage1_time_ms,

    -- Stage 3 metrics
    COUNT(*) FILTER (WHERE stage3_passed = true) as stage3_passed,
    AVG(stage3_probability) FILTER (WHERE stage3_passed = true) as avg_stage3_prob,

    -- Stage 4 metrics
    COUNT(*) FILTER (WHERE stage4_passed = true) as stage4_passed,
    AVG(stage4_dynamic_threshold) as avg_dynamic_threshold,

    -- Final metrics
    COUNT(*) FILTER (WHERE final_decision = true) as signals_to_trade,
    COUNT(*) FILTER (WHERE final_decision = true AND actual_outcome = true) as wins,
    COUNT(*) FILTER (WHERE final_decision = true AND actual_outcome = false) as losses,

    -- Win rate
    CASE
        WHEN COUNT(*) FILTER (WHERE final_decision = true AND actual_outcome IS NOT NULL) > 0
        THEN COUNT(*) FILTER (WHERE final_decision = true AND actual_outcome = true)::NUMERIC /
             COUNT(*) FILTER (WHERE final_decision = true AND actual_outcome IS NOT NULL)
        ELSE NULL
    END as win_rate,

    -- Pass rate at each stage
    COUNT(*) FILTER (WHERE stage1_passed = true)::NUMERIC / NULLIF(COUNT(*), 0) as stage1_pass_rate,
    COUNT(*) FILTER (WHERE stage3_passed = true)::NUMERIC / NULLIF(COUNT(*) FILTER (WHERE stage1_passed = true), 0) as stage3_pass_rate,
    COUNT(*) FILTER (WHERE stage4_passed = true)::NUMERIC / NULLIF(COUNT(*) FILTER (WHERE stage3_passed = true), 0) as stage4_pass_rate,
    COUNT(*) FILTER (WHERE final_decision = true)::NUMERIC / NULLIF(COUNT(*), 0) as final_pass_rate,

    -- Processing time
    AVG(total_processing_time_ms) as avg_total_time_ms

FROM ml.multistage_predictions
GROUP BY DATE(predicted_at), signal_type
ORDER BY prediction_date DESC, signal_type;

-- 9. Функция для мониторинга и алертов
CREATE OR REPLACE FUNCTION ml.check_system_health()
RETURNS TABLE (
    check_name VARCHAR,
    status VARCHAR,
    details TEXT
) AS $$
BEGIN
    -- Check 1: Models availability
    RETURN QUERY
    SELECT
        'Models Availability'::VARCHAR,
        CASE
            WHEN COUNT(*) >= 6 THEN 'OK'::VARCHAR
            ELSE 'WARNING'::VARCHAR
        END,
        FORMAT('Active models: %s (need at least 6)', COUNT(*))::TEXT
    FROM ml.model_registry
    WHERE is_active = true;

    -- Check 2: Recent predictions
    RETURN QUERY
    SELECT
        'Recent Predictions'::VARCHAR,
        CASE
            WHEN COUNT(*) > 0 THEN 'OK'::VARCHAR
            ELSE 'ERROR'::VARCHAR
        END,
        FORMAT('Predictions in last hour: %s', COUNT(*))::TEXT
    FROM ml.multistage_predictions
    WHERE predicted_at >= NOW() - INTERVAL '1 hour';

    -- Check 3: Win rate
    RETURN QUERY
    SELECT
        'Win Rate'::VARCHAR,
        CASE
            WHEN AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) >= 0.70 THEN 'OK'::VARCHAR
            WHEN AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) >= 0.60 THEN 'WARNING'::VARCHAR
            ELSE 'ERROR'::VARCHAR
        END,
        FORMAT('24h win rate: %.1f%%',
               AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) * 100)::TEXT
    FROM ml.multistage_predictions
    WHERE final_decision = true
        AND actual_outcome IS NOT NULL
        AND predicted_at >= NOW() - INTERVAL '24 hours';

    -- Check 4: Pass rate
    RETURN QUERY
    SELECT
        'Signal Pass Rate'::VARCHAR,
        CASE
            WHEN AVG(CASE WHEN final_decision THEN 1 ELSE 0 END) BETWEEN 0.05 AND 0.15 THEN 'OK'::VARCHAR
            ELSE 'WARNING'::VARCHAR
        END,
        FORMAT('Final pass rate: %.1f%% (target: 5-15%%)',
               AVG(CASE WHEN final_decision THEN 1 ELSE 0 END) * 100)::TEXT
    FROM ml.multistage_predictions
    WHERE predicted_at >= NOW() - INTERVAL '24 hours';

END;
$$ LANGUAGE plpgsql;

-- 10. Автоматическое обновление метрик производительности
CREATE OR REPLACE FUNCTION ml.update_performance_metrics()
RETURNS void AS $$
BEGIN
    -- Обновляем live_win_rate в model_registry
    UPDATE ml.model_registry mr
    SET
        live_win_rate = subq.win_rate,
        total_predictions = subq.total,
        successful_predictions = subq.wins,
        last_used_at = NOW()
    FROM (
        SELECT
            stage4_model_id as model_id,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE actual_outcome = true) as wins,
            AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) as win_rate
        FROM ml.multistage_predictions
        WHERE actual_outcome IS NOT NULL
            AND predicted_at >= NOW() - INTERVAL '7 days'
        GROUP BY stage4_model_id
    ) subq
    WHERE mr.id = subq.model_id;

    -- Записываем метрики в performance_monitor
    INSERT INTO ml.performance_monitor (
        model_id,
        monitoring_window,
        total_signals,
        signals_passed,
        true_positives,
        false_positives,
        win_rate,
        pass_rate,
        avg_probability,
        target_win_rate
    )
    SELECT
        mr.id,
        '24h',
        COUNT(*),
        COUNT(*) FILTER (WHERE mp.final_decision = true),
        COUNT(*) FILTER (WHERE mp.final_decision = true AND mp.actual_outcome = true),
        COUNT(*) FILTER (WHERE mp.final_decision = true AND mp.actual_outcome = false),
        AVG(CASE WHEN mp.actual_outcome THEN 1 ELSE 0 END) FILTER (WHERE mp.final_decision = true),
        AVG(CASE WHEN mp.final_decision THEN 1 ELSE 0 END),
        AVG(mp.stage4_probability) FILTER (WHERE mp.final_decision = true),
        CASE
            WHEN mr.signal_type = 'BUY' THEN 0.85
            ELSE 0.70
        END
    FROM ml.model_registry mr
    LEFT JOIN ml.multistage_predictions mp ON mp.stage4_model_id = mr.id
    WHERE mp.predicted_at >= NOW() - INTERVAL '24 hours'
        AND mr.is_active = true
    GROUP BY mr.id, mr.signal_type
    ON CONFLICT (model_id, monitoring_window, calculated_at) DO NOTHING;

END;
$$ LANGUAGE plpgsql;

-- Создаем задания для автоматизации (если используется pg_cron)
-- SELECT cron.schedule('update-performance-metrics', '*/30 * * * *', 'SELECT ml.update_performance_metrics();');
-- SELECT cron.schedule('calibrate-thresholds', '0 * * * *', 'SELECT ml.calibrate_threshold(id, 0.85) FROM ml.model_registry WHERE is_active = true;');