"""
Production Predictor V2 - Processes active signals from scoring_history.
Reads active signals, makes predictions, saves to database, marks as processed.
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import logging
from dotenv import load_dotenv
import os
import hashlib
import json

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionPredictorV2:
    """Processes active signals from scoring_history table."""

    def __init__(self):
        """Initialize predictor with database connection and models."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_columns = {}
        self.model_versions = {}

        self._load_models()

    def _load_models(self):
        """Load trained ML models."""
        for signal_type in ['BUY', 'SELL']:
            model_path = f'models/{signal_type.lower()}_adaptive_model.pkl'

            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    self.models[signal_type] = model_data['model']
                    self.scalers[signal_type] = model_data['scaler']
                    self.thresholds[signal_type] = model_data['threshold']
                    self.feature_columns[signal_type] = model_data['feature_columns']

                    # Create version string
                    window_days = model_data.get('window_days', 7)
                    model_date = model_data.get('timestamp', datetime.now().isoformat())[:10]
                    self.model_versions[signal_type] = f"adaptive_{window_days}d_{model_date}"

                    logger.info(f"✅ Loaded {signal_type} model: {self.model_versions[signal_type]}, "
                                f"threshold={self.thresholds[signal_type]:.3f}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {signal_type} model: {e}")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")

    def get_active_signals(self):
        """Fetch active signals from database using the provided query."""
        query = """
        WITH active_signals AS (
            SELECT
                sh.id,
                sh.timestamp,
                sh.trading_pair_id,
                sh.pair_symbol,
                sh.indicator_score,
                sh.pattern_score,
                sh.combination_score,
                sh.total_score,
                sh.patterns_details,
                sh.combinations_details,
                sh.created_at,
                CASE
                    WHEN sh.total_score > 0 THEN 'BUY'
                    ELSE 'SELL'
                END AS signal_type,
                CASE
                    WHEN abs(sh.total_score) >= 100 THEN 'VERY_STRONG'
                    WHEN abs(sh.total_score) >= 50 THEN 'STRONG'
                    WHEN abs(sh.total_score) >= 20 THEN 'MODERATE'
                    ELSE 'WEAK'
                END AS signal_strength,
                (
                    SELECT mr.regime
                    FROM fas.market_regime mr
                    WHERE mr.timeframe = '4h'::fas.timeframe_enum AND mr.timestamp <= sh.timestamp
                    ORDER BY mr.timestamp DESC
                    LIMIT 1
                ) AS market_regime,
                public.is_meme_coin(sh.trading_pair_id) AS is_meme,
                ind.close_price,
                ind.price_change_pct,
                ind.buy_ratio,
                ind.buy_ratio_weighted,
                ind.normalized_imbalance,
                ind.smoothed_imbalance,
                ind.volume_zscore,
                ind.cvd_delta,
                ind.cvd_cumulative,
                ind.oi_delta_pct,
                ind.funding_rate_avg,
                ind.rsi,
                ind.rs_value,
                ind.rs_momentum,
                ind.atr,
                ind.macd_line,
                ind.macd_signal,
                ind.macd_histogram,
                poc.poc_24h,
                poc.poc_7d,
                poc.poc_30d,
                poc.volume_24h AS poc_volume_24h,
                poc.volume_7d AS poc_volume_7d,
                poc.calculated_at AS poc_calculated_at
            FROM (SELECT * FROM fas.scoring_history WHERE is_active = true) sh
            LEFT JOIN LATERAL (
                SELECT *
                FROM fas.indicators i
                WHERE i.trading_pair_id = sh.trading_pair_id
                  AND i.timeframe = '15m'::fas.timeframe_enum
                  AND i.timestamp <= sh.timestamp
                ORDER BY i.timestamp DESC
                LIMIT 1
            ) ind ON true
            LEFT JOIN LATERAL (
                SELECT *
                FROM fas.poc_levels p
                WHERE p.trading_pair_id = sh.trading_pair_id
                  AND p.calculated_at = date_trunc('hour', sh.timestamp)
                ORDER BY p.calculated_at DESC
                LIMIT 1
            ) poc ON true
            WHERE NOT public.is_stablecoin_pair(sh.trading_pair_id)
        ),
        patterns_expanded AS (
            SELECT
                id,
                (patterns_details -> 0) ->> 'pattern' AS pattern_1_name,
                ((patterns_details -> 0) ->> 'impact')::numeric AS pattern_1_impact,
                ((patterns_details -> 0) ->> 'confidence')::numeric AS pattern_1_confidence,
                (patterns_details -> 1) ->> 'pattern' AS pattern_2_name,
                ((patterns_details -> 1) ->> 'impact')::numeric AS pattern_2_impact,
                ((patterns_details -> 1) ->> 'confidence')::numeric AS pattern_2_confidence,
                (patterns_details -> 2) ->> 'pattern' AS pattern_3_name,
                ((patterns_details -> 2) ->> 'impact')::numeric AS pattern_3_impact,
                ((patterns_details -> 2) ->> 'confidence')::numeric AS pattern_3_confidence,
                jsonb_array_length(patterns_details) AS pattern_count,
                CASE WHEN patterns_details::text LIKE '%DISTRIBUTION%' THEN 1 ELSE 0 END AS has_distribution,
                CASE WHEN patterns_details::text LIKE '%ACCUMULATION%' THEN 1 ELSE 0 END AS has_accumulation,
                CASE WHEN patterns_details::text LIKE '%VOLUME_ANOMALY%' THEN 1 ELSE 0 END AS has_volume_anomaly,
                CASE WHEN patterns_details::text LIKE '%MOMENTUM_EXHAUSTION%' THEN 1 ELSE 0 END AS has_momentum_exhaustion,
                CASE WHEN patterns_details::text LIKE '%OI_EXPLOSION%' THEN 1 ELSE 0 END AS has_oi_explosion,
                CASE WHEN patterns_details::text LIKE '%SQUEEZE_IGNITION%' THEN 1 ELSE 0 END AS has_squeeze_ignition,
                CASE WHEN patterns_details::text LIKE '%CVD_PRICE_DIVERGENCE%' THEN 1 ELSE 0 END AS has_cvd_divergence
            FROM active_signals
        ),
        combinations_expanded AS (
            SELECT
                id,
                (combinations_details -> 0) ->> 'combination_name' AS combo_1_name,
                ((combinations_details -> 0) ->> 'score')::numeric AS combo_1_score,
                ((combinations_details -> 0) ->> 'confidence')::numeric AS combo_1_confidence,
                (combinations_details -> 1) ->> 'combination_name' AS combo_2_name,
                ((combinations_details -> 1) ->> 'score')::numeric AS combo_2_score,
                ((combinations_details -> 1) ->> 'confidence')::numeric AS combo_2_confidence,
                jsonb_array_length(combinations_details) AS combo_count,
                CASE WHEN combinations_details::text LIKE '%VOLUME_DISTRIBUTION%' THEN 1 ELSE 0 END AS has_volume_distribution,
                CASE WHEN combinations_details::text LIKE '%VOLUME_ACCUMULATION%' THEN 1 ELSE 0 END AS has_volume_accumulation,
                CASE WHEN combinations_details::text LIKE '%INSTITUTIONAL_SURGE%' THEN 1 ELSE 0 END AS has_institutional_surge,
                CASE WHEN combinations_details::text LIKE '%SQUEEZE_MOMENTUM%' THEN 1 ELSE 0 END AS has_squeeze_momentum,
                CASE WHEN combinations_details::text LIKE '%SMART_ACCUMULATION%' THEN 1 ELSE 0 END AS has_smart_accumulation
            FROM active_signals
        )
        SELECT
            so.id,
            so.timestamp,
            so.trading_pair_id,
            so.pair_symbol,
            so.indicator_score,
            so.pattern_score,
            so.combination_score,
            so.total_score,
            so.signal_type,
            so.signal_strength,
            CASE so.signal_strength
                WHEN 'VERY_STRONG' THEN 4
                WHEN 'STRONG' THEN 3
                WHEN 'MODERATE' THEN 2
                WHEN 'WEAK' THEN 1
                ELSE 0
            END AS strength_numeric,
            so.market_regime,
            so.is_meme,
            so.close_price,
            so.price_change_pct,
            so.buy_ratio,
            so.buy_ratio_weighted,
            so.normalized_imbalance,
            so.smoothed_imbalance,
            so.volume_zscore,
            so.cvd_delta,
            so.cvd_cumulative,
            so.oi_delta_pct,
            so.funding_rate_avg,
            so.rsi,
            CASE
                WHEN so.rsi > 70 THEN 1
                WHEN so.rsi < 30 THEN -1
                ELSE 0
            END AS rsi_zone,
            so.rs_value,
            so.rs_momentum,
            so.atr,
            so.atr / NULLIF(so.close_price, 0) * 100 AS atr_pct,
            so.macd_line,
            so.macd_signal,
            so.macd_histogram,
            CASE WHEN so.poc_24h > 0 THEN (so.close_price - so.poc_24h) / so.poc_24h * 100 ELSE NULL END AS price_to_poc_24h_pct,
            CASE WHEN so.poc_7d > 0 THEN (so.close_price - so.poc_7d) / so.poc_7d * 100 ELSE NULL END AS price_to_poc_7d_pct,
            CASE WHEN so.poc_30d > 0 THEN (so.close_price - so.poc_30d) / so.poc_30d * 100 ELSE NULL END AS price_to_poc_30d_pct,
            so.poc_volume_24h,
            so.poc_volume_7d,
            pe.pattern_1_name,
            pe.pattern_1_impact,
            pe.pattern_1_confidence,
            pe.pattern_2_name,
            pe.pattern_2_impact,
            pe.pattern_2_confidence,
            pe.pattern_3_name,
            pe.pattern_3_impact,
            pe.pattern_3_confidence,
            pe.pattern_count,
            pe.has_distribution,
            pe.has_accumulation,
            pe.has_volume_anomaly,
            pe.has_momentum_exhaustion,
            pe.has_oi_explosion,
            pe.has_squeeze_ignition,
            pe.has_cvd_divergence,
            ce.combo_1_name,
            ce.combo_1_score,
            ce.combo_1_confidence,
            ce.combo_2_name,
            ce.combo_2_score,
            ce.combo_2_confidence,
            ce.combo_count,
            ce.has_volume_distribution,
            ce.has_volume_accumulation,
            ce.has_institutional_surge,
            ce.has_squeeze_momentum,
            ce.has_smart_accumulation
        FROM active_signals so
        LEFT JOIN patterns_expanded pe ON so.id = pe.id
        LEFT JOIN combinations_expanded ce ON so.id = ce.id
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                df = pd.read_sql(query, conn)

            logger.info(f"📥 Fetched {len(df)} active signals from database")
            if len(df) > 0:
                by_type = df['signal_type'].value_counts()
                logger.info(f"   BUY: {by_type.get('BUY', 0)}, SELL: {by_type.get('SELL', 0)}")

            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch active signals: {e}")
            return pd.DataFrame()

    def prepare_features(self, df, signal_type):
        """Prepare features for prediction - matching training preparation."""
        df_proc = df.copy()

        # Remove columns not needed for prediction
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                       'signal_type', 'signal_strength']

        for col in remove_cols:
            if col in df_proc.columns:
                df_proc = df_proc.drop(columns=[col])

        # Fix extreme POC volumes
        for col in ['poc_volume_7d', 'poc_volume_24h']:
            if col in df_proc.columns:
                q99 = df_proc[col].quantile(0.99) if len(df_proc) > 100 else df_proc[col].max()
                df_proc[col] = df_proc[col].clip(upper=q99)
                df_proc[f'{col}_log'] = np.log1p(df_proc[col])

                if df_proc[col].max() / (df_proc[col].quantile(0.95) + 1) > 100:
                    df_proc = df_proc.drop(columns=[col])

        # Add time features
        df_proc['timestamp_hour'] = datetime.now().hour
        df_proc['timestamp_day'] = datetime.now().weekday()

        # Add recent performance features (critical for model)
        # For production, we estimate these based on recent market conditions
        if signal_type == 'BUY':
            df_proc['recent_avg'] = 0.38  # Recent average from validation
            df_proc['recent_std'] = 0.15
        else:
            df_proc['recent_avg'] = 0.60
            df_proc['recent_std'] = 0.20

        # Market regime features
        if 'market_regime' in df_proc.columns:
            df_proc['regime_bull'] = (df_proc['market_regime'] == 'BULL').astype(float)
            df_proc['regime_bear'] = (df_proc['market_regime'] == 'BEAR').astype(float)
            df_proc['regime_neutral'] = (df_proc['market_regime'] == 'NEUTRAL').astype(float)

            if signal_type == 'BUY':
                df_proc['good_alignment'] = df_proc['regime_bull']
                df_proc['bad_alignment'] = df_proc['regime_bear']
            else:
                df_proc['good_alignment'] = df_proc['regime_bear']
                df_proc['bad_alignment'] = df_proc['regime_bull']

        # Handle categorical columns
        categorical_cols = df_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_proc[col] = pd.Categorical(df_proc[col].fillna('unknown')).codes

        # Ensure all required features are present
        for feat in self.feature_columns[signal_type]:
            if feat not in df_proc.columns:
                df_proc[feat] = 0

        # Select and scale features
        X = df_proc[self.feature_columns[signal_type]].fillna(0)
        X_scaled = pd.DataFrame(
            self.scalers[signal_type].transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_scaled

    def make_predictions(self, df):
        """Make predictions for all signals."""
        predictions = []

        for signal_type in ['BUY', 'SELL']:
            if signal_type not in self.models:
                logger.warning(f"⚠️ No model for {signal_type}")
                continue

            # Filter signals by type
            type_signals = df[df['signal_type'] == signal_type].copy()

            if len(type_signals) == 0:
                continue

            logger.info(f"🔮 Processing {len(type_signals)} {signal_type} signals...")

            # Prepare features
            X_scaled = self.prepare_features(type_signals, signal_type)

            # Make predictions
            model = self.models[signal_type]
            threshold = self.thresholds[signal_type]

            # Get probabilities
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]

            # Apply threshold
            y_pred = (y_pred_proba >= threshold).astype(bool)

            # Calculate confidence
            confidence = []
            for prob, thresh in zip(y_pred_proba, [threshold] * len(y_pred_proba)):
                if prob > thresh + 0.15:
                    confidence.append('HIGH')
                elif prob > thresh:
                    confidence.append('MEDIUM')
                else:
                    confidence.append('LOW')

            # Create prediction records
            for idx, (_, signal) in enumerate(type_signals.iterrows()):
                # Create features hash for debugging
                features_dict = X_scaled.iloc[idx].to_dict()
                features_hash = hashlib.md5(
                    json.dumps(features_dict, sort_keys=True).encode()
                ).hexdigest()

                predictions.append({
                    'signal_id': signal['id'],
                    'signal_timestamp': signal['timestamp'],
                    'trading_pair_id': signal['trading_pair_id'],
                    'pair_symbol': signal['pair_symbol'],
                    'signal_type': signal_type,
                    'total_score': signal['total_score'],
                    'market_regime': signal['market_regime'],
                    'model_version': self.model_versions[signal_type],
                    'prediction_probability': float(y_pred_proba[idx]),
                    'threshold_used': float(threshold),
                    'prediction': bool(y_pred[idx]),
                    'confidence_level': confidence[idx],
                    'features_hash': features_hash[:16],  # Short hash
                    'model_metadata': {
                        'features_count': len(self.feature_columns[signal_type]),
                        'regime': signal['market_regime'],
                        'strength': signal.get('signal_strength', 'UNKNOWN')
                    }
                })

        logger.info(f"📊 Generated {len(predictions)} predictions")
        return predictions

    def save_predictions(self, predictions):
        """Save predictions to database."""
        if not predictions:
            logger.warning("No predictions to save")
            return []

        saved_ids = []

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                for pred in predictions:
                    try:
                        cur.execute("""
                            INSERT INTO ml.signal_predictions (
                                signal_id, signal_timestamp, trading_pair_id, pair_symbol,
                                signal_type, total_score, market_regime,
                                model_version, prediction_probability, threshold_used,
                                prediction, confidence_level, features_hash, model_metadata
                            ) VALUES (
                                %(signal_id)s, %(signal_timestamp)s, %(trading_pair_id)s, %(pair_symbol)s,
                                %(signal_type)s, %(total_score)s, %(market_regime)s,
                                %(model_version)s, %(prediction_probability)s, %(threshold_used)s,
                                %(prediction)s, %(confidence_level)s, %(features_hash)s, %(model_metadata)s
                            )
                            ON CONFLICT (signal_id) DO UPDATE SET
                                prediction_probability = EXCLUDED.prediction_probability,
                                prediction = EXCLUDED.prediction,
                                predicted_at = NOW()
                            RETURNING id, signal_id
                        """, {
                            **pred,
                            'model_metadata': json.dumps(pred['model_metadata'])
                        })

                        result = cur.fetchone()
                        if result:
                            saved_ids.append(result[1])  # signal_id

                    except Exception as e:
                        logger.error(f"Failed to save prediction for signal {pred['signal_id']}: {e}")
                        conn.rollback()
                        continue

                conn.commit()

        logger.info(f"💾 Saved {len(saved_ids)} predictions to database")
        return saved_ids

    def mark_signals_processed(self, signal_ids):
        """Mark processed signals as inactive."""
        if not signal_ids:
            return

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Convert to tuple for SQL IN clause
                ids_tuple = tuple(signal_ids)

                cur.execute("""
                    UPDATE fas.scoring_history
                    SET is_active = false
                    WHERE id IN %s
                """, (ids_tuple,))

                updated = cur.rowcount
                conn.commit()

        logger.info(f"✅ Marked {updated} signals as processed (is_active=false)")

    def get_prediction_summary(self, predictions):
        """Generate summary of predictions."""
        if not predictions:
            return

        df = pd.DataFrame(predictions)

        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)

        # Overall stats
        total_trade = df['prediction'].sum()
        total_skip = len(df) - total_trade

        logger.info(f"\nTotal signals: {len(df)}")
        logger.info(f"  📈 Trade: {total_trade} ({total_trade / len(df) * 100:.1f}%)")
        logger.info(f"  ⏭️ Skip: {total_skip} ({total_skip / len(df) * 100:.1f}%)")

        # By signal type
        for signal_type in ['BUY', 'SELL']:
            type_df = df[df['signal_type'] == signal_type]
            if len(type_df) > 0:
                trades = type_df['prediction'].sum()
                avg_prob = type_df[type_df['prediction']]['prediction_probability'].mean() if trades > 0 else 0

                logger.info(f"\n{signal_type} Signals:")
                logger.info(f"  Total: {len(type_df)}")
                logger.info(f"  Trade: {trades} ({trades / len(type_df) * 100:.1f}%)")
                logger.info(f"  Avg probability: {avg_prob:.3f}")

                # Top signals
                if trades > 0:
                    top_signals = type_df[type_df['prediction']].nlargest(
                        min(3, trades), 'prediction_probability'
                    )
                    logger.info(f"  Top signals:")
                    for _, sig in top_signals.iterrows():
                        logger.info(f"    {sig['pair_symbol']}: {sig['prediction_probability']:.3f} "
                                    f"({sig['confidence_level']})")

    def run(self):
        """Main execution flow."""
        logger.info("\n" + "=" * 60)
        logger.info("PRODUCTION PREDICTOR V2 - PROCESSING ACTIVE SIGNALS")
        logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 60)

        # 1. Get active signals
        active_signals = self.get_active_signals()

        if len(active_signals) == 0:
            logger.info("✅ No active signals to process")
            return

        # 2. Make predictions
        predictions = self.make_predictions(active_signals)

        # 3. Show summary
        self.get_prediction_summary(predictions)

        # 4. Save predictions to database
        saved_signal_ids = self.save_predictions(predictions)

        # 5. Mark signals as processed
        self.mark_signals_processed(saved_signal_ids)

        # 6. Export to CSV for review (optional)
        if predictions:
            df_pred = pd.DataFrame(predictions)
            csv_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_pred.to_csv(csv_file, index=False)
            logger.info(f"📄 Predictions exported to {csv_file}")

        logger.info("\n✅ Processing complete!")

        # Return predictions for potential further processing
        return predictions


def main():
    """Run the production predictor."""
    predictor = ProductionPredictorV2()
    predictions = predictor.run()

    # Optional: Show signals to trade
    if predictions:
        trades = [p for p in predictions if p['prediction']]
        if trades:
            logger.info(f"\n🎯 {len(trades)} SIGNALS TO TRADE NOW:")
            for trade in sorted(trades, key=lambda x: x['prediction_probability'], reverse=True)[:10]:
                logger.info(f"  {trade['signal_type']} {trade['pair_symbol']}: "
                            f"prob={trade['prediction_probability']:.3f} "
                            f"({trade['confidence_level']})")


if __name__ == "__main__":
    main()