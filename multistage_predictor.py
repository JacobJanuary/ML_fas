"""
Multi-Stage Filtering System for High Win Rate Trading
=======================================================
Implements 4-stage filtering to achieve 85%+ win rate
while trading only 5-10% of top signals.
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from a single stage of filtering."""
    passed: bool
    probability: float
    threshold: float
    model_id: int
    processing_time_ms: int
    metadata: Dict = None


class MultiStagePredictor:
    """
    Multi-stage filtering system for ultra-high precision trading.

    Architecture:
        Stage 1: Quick Filter (70% reduction)
        Stage 2: Regime Detection
        Stage 3: Specialized Models (85% reduction)
        Stage 4: Precision Filter + Dynamic Calibration (50% reduction)

    Target: 5-10% pass rate, 85%+ win rate
    """

    def __init__(self):
        """Initialize the multi-stage prediction system."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Model cache
        self.models = {}
        self.scalers = {}
        self.thresholds = {}

        # Performance targets
        self.target_win_rates = {
            'BUY': 0.85,  # 85% win rate for BUY
            'SELL': 0.70  # 70% win rate for SELL
        }

        # Load all models
        self._load_models()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _load_models(self):
        """Load all models from registry."""
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, model_name, model_path, threshold
                    FROM ml.model_registry
                    WHERE is_active = true
                """)

                for row in cur.fetchall():
                    model_id = row['id']
                    model_path = row['model_path']

                    if model_path and os.path.exists(model_path):
                        try:
                            model_data = joblib.load(model_path)
                            self.models[model_id] = model_data.get('model')
                            self.scalers[model_id] = model_data.get('scaler')
                            self.thresholds[model_id] = row['threshold'] or model_data.get('threshold', 0.5)

                            logger.info(f"Loaded model {row['model_name']} (ID: {model_id})")
                        except Exception as e:
                            logger.error(f"Failed to load model {model_id}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

    def _get_model_for_stage(self, stage: str, signal_type: str, regime: str = None) -> Optional[int]:
        """Get best model for specific stage and conditions."""
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Use function from SQL to select best model
                cur.execute(
                    "SELECT ml.select_best_model(%s, %s, %s)",
                    (signal_type, regime, stage)
                )
                model_id = cur.fetchone()[0]

                return model_id

    def stage1_quick_filter(self, features: pd.DataFrame, signal_type: str) -> StageResult:
        """
        Stage 1: Quick Filter
        Fast XGBoost model to eliminate obvious bad signals.
        Target: Remove 70% of signals, <10ms processing.
        """
        start_time = time.time()

        # Get quick filter model
        model_id = self._get_model_for_stage('quick_filter', signal_type)

        if not model_id or model_id not in self.models:
            # Fallback to simple rules if no model
            score_threshold = 10 if signal_type == 'BUY' else -10
            passed = features['total_score'].iloc[0] > score_threshold if signal_type == 'BUY' else \
            features['total_score'].iloc[0] < score_threshold

            return StageResult(
                passed=passed,
                probability=0.5,
                threshold=0.3,
                model_id=0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # Make prediction
        model = self.models[model_id]
        scaler = self.scalers.get(model_id)

        # Prepare features (subset for speed)
        quick_features = ['total_score', 'rsi', 'volume_zscore', 'price_change_pct']
        X = features[quick_features].fillna(0)

        if scaler:
            X = scaler.transform(X)

        probability = model.predict_proba(X)[0, 1]
        threshold = 0.3  # Low threshold for quick filter

        return StageResult(
            passed=probability >= threshold,
            probability=float(probability),
            threshold=threshold,
            model_id=model_id,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

    def stage2_detect_regime(self, features: pd.DataFrame) -> Tuple[str, float]:
        """
        Stage 2: Market Regime Detection
        Determines current market regime for model selection.
        """
        # Get latest market regime from database
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT regime, 
                           CASE regime
                               WHEN 'BULL' THEN 0.8
                               WHEN 'BEAR' THEN 0.8
                               ELSE 0.6
                           END as confidence
                    FROM fas.market_regime
                    WHERE timeframe = '4h'::fas.timeframe_enum
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)

                result = cur.fetchone()
                if result:
                    return result[0], result[1]

        # Fallback: Simple regime detection based on indicators
        rsi = features['rsi'].iloc[0] if 'rsi' in features else 50

        if rsi > 65:
            return 'BULL', 0.7
        elif rsi < 35:
            return 'BEAR', 0.7
        else:
            return 'NEUTRAL', 0.6

    def stage3_specialized_model(self, features: pd.DataFrame, signal_type: str, regime: str) -> StageResult:
        """
        Stage 3: Specialized Model
        Regime-specific model for better accuracy.
        """
        start_time = time.time()

        # Get specialized model for this regime
        model_id = self._get_model_for_stage('regime_specific', signal_type, regime)

        if not model_id or model_id not in self.models:
            # Fallback to universal model
            model_id = self._get_model_for_stage('regime_specific', signal_type, None)

        if not model_id or model_id not in self.models:
            return StageResult(
                passed=False,
                probability=0.0,
                threshold=0.5,
                model_id=0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # Make prediction with full features
        model = self.models[model_id]
        scaler = self.scalers.get(model_id)

        X = features.fillna(0)
        if scaler:
            X = scaler.transform(X)

        probability = model.predict_proba(X)[0, 1]
        threshold = self.thresholds.get(model_id, 0.6)

        return StageResult(
            passed=probability >= threshold,
            probability=float(probability),
            threshold=threshold,
            model_id=model_id,
            processing_time_ms=int((time.time() - start_time) * 1000),
            metadata={'regime': regime}
        )

    def stage4_precision_filter(self, features: pd.DataFrame, signal_type: str,
                                previous_probabilities: List[float]) -> StageResult:
        """
        Stage 4: Precision Filter with Dynamic Calibration
        Final high-precision model with dynamically calibrated threshold.
        """
        start_time = time.time()

        # Get precision model
        model_id = self._get_model_for_stage('precision', signal_type)

        if not model_id or model_id not in self.models:
            # Use weighted average of previous stages as fallback
            avg_prob = np.mean(previous_probabilities)

            return StageResult(
                passed=avg_prob >= 0.7,
                probability=avg_prob,
                threshold=0.7,
                model_id=0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # Make prediction
        model = self.models[model_id]
        scaler = self.scalers.get(model_id)

        # Add ensemble features (probabilities from previous stages)
        ensemble_features = features.copy()
        ensemble_features['stage1_prob'] = previous_probabilities[0]
        ensemble_features['stage3_prob'] = previous_probabilities[1]
        ensemble_features['prob_std'] = np.std(previous_probabilities)
        ensemble_features['prob_mean'] = np.mean(previous_probabilities)

        X = ensemble_features.fillna(0)
        if scaler:
            X = scaler.transform(X)

        probability = model.predict_proba(X)[0, 1]

        # Get dynamically calibrated threshold
        dynamic_threshold = self._get_dynamic_threshold(model_id, signal_type)

        return StageResult(
            passed=probability >= dynamic_threshold,
            probability=float(probability),
            threshold=dynamic_threshold,
            model_id=model_id,
            processing_time_ms=int((time.time() - start_time) * 1000),
            metadata={'calibrated': True}
        )

    def _get_dynamic_threshold(self, model_id: int, signal_type: str) -> float:
        """Get dynamically calibrated threshold for model."""
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Get latest calibration
                cur.execute("""
                    SELECT selected_threshold
                    FROM ml.threshold_calibration
                    WHERE model_id = %s
                        AND is_active = true
                    ORDER BY calibration_timestamp DESC
                    LIMIT 1
                """, (model_id,))

                result = cur.fetchone()
                if result:
                    return float(result[0])

                # Fallback to target-based threshold
                target_wr = self.target_win_rates[signal_type]

                # Calibrate based on recent predictions
                cur.execute(
                    "SELECT ml.calibrate_threshold(%s, %s, 48)",
                    (model_id, target_wr)
                )

                calibrated = cur.fetchone()[0]
                if calibrated:
                    return float(calibrated)

        # Ultimate fallback
        return 0.7 if signal_type == 'BUY' else 0.6

    def process_signal(self, signal_data: Dict) -> Dict:
        """
        Process a single signal through all stages.

        Returns:
            Dict with prediction results and metadata
        """
        total_start = time.time()

        signal_id = signal_data['id']
        signal_type = signal_data['signal_type']

        # Prepare features
        features = pd.DataFrame([signal_data])

        # Stage 1: Quick Filter
        stage1_result = self.stage1_quick_filter(features, signal_type)

        if not stage1_result.passed:
            return self._create_result(
                signal_data,
                final_decision=False,
                stage1_result=stage1_result,
                total_time_ms=int((time.time() - total_start) * 1000)
            )

        # Stage 2: Regime Detection
        regime, regime_confidence = self.stage2_detect_regime(features)

        # Stage 3: Specialized Model
        stage3_result = self.stage3_specialized_model(features, signal_type, regime)

        if not stage3_result.passed:
            return self._create_result(
                signal_data,
                final_decision=False,
                stage1_result=stage1_result,
                stage3_result=stage3_result,
                regime=regime,
                regime_confidence=regime_confidence,
                total_time_ms=int((time.time() - total_start) * 1000)
            )

        # Stage 4: Precision Filter
        previous_probs = [stage1_result.probability, stage3_result.probability]
        stage4_result = self.stage4_precision_filter(features, signal_type, previous_probs)

        # Calculate final confidence
        all_probs = previous_probs + [stage4_result.probability]
        final_confidence = np.mean(all_probs) * min(all_probs) / max(all_probs)

        return self._create_result(
            signal_data,
            final_decision=stage4_result.passed,
            final_confidence=final_confidence,
            stage1_result=stage1_result,
            stage3_result=stage3_result,
            stage4_result=stage4_result,
            regime=regime,
            regime_confidence=regime_confidence,
            total_time_ms=int((time.time() - total_start) * 1000)
        )

    def _create_result(self, signal_data: Dict, **kwargs) -> Dict:
        """Create result dictionary for database storage."""
        return {
            'signal_id': signal_data['id'],
            'signal_timestamp': signal_data['timestamp'],
            'trading_pair_id': signal_data['trading_pair_id'],
            'pair_symbol': signal_data['pair_symbol'],
            'signal_type': signal_data['signal_type'],

            'stage1_model_id': kwargs.get('stage1_result', StageResult(False, 0, 0, 0, 0)).model_id,
            'stage1_probability': kwargs.get('stage1_result', StageResult(False, 0, 0, 0, 0)).probability,
            'stage1_passed': kwargs.get('stage1_result', StageResult(False, 0, 0, 0, 0)).passed,
            'stage1_time_ms': kwargs.get('stage1_result', StageResult(False, 0, 0, 0, 0)).processing_time_ms,

            'detected_regime': kwargs.get('regime'),
            'regime_confidence': kwargs.get('regime_confidence'),

            'stage3_model_id': kwargs.get('stage3_result', StageResult(False, 0, 0, 0, 0)).model_id,
            'stage3_probability': kwargs.get('stage3_result', StageResult(False, 0, 0, 0, 0)).probability,
            'stage3_threshold': kwargs.get('stage3_result', StageResult(False, 0, 0, 0, 0)).threshold,
            'stage3_passed': kwargs.get('stage3_result', StageResult(False, 0, 0, 0, 0)).passed,
            'stage3_time_ms': kwargs.get('stage3_result', StageResult(False, 0, 0, 0, 0)).processing_time_ms,

            'stage4_model_id': kwargs.get('stage4_result', StageResult(False, 0, 0, 0, 0)).model_id,
            'stage4_probability': kwargs.get('stage4_result', StageResult(False, 0, 0, 0, 0)).probability,
            'stage4_dynamic_threshold': kwargs.get('stage4_result', StageResult(False, 0, 0, 0, 0)).threshold,
            'stage4_passed': kwargs.get('stage4_result', StageResult(False, 0, 0, 0, 0)).passed,
            'stage4_time_ms': kwargs.get('stage4_result', StageResult(False, 0, 0, 0, 0)).processing_time_ms,

            'final_decision': kwargs.get('final_decision', False),
            'final_confidence': kwargs.get('final_confidence', 0.0),
            'total_processing_time_ms': kwargs.get('total_time_ms', 0)
        }

    def process_batch(self, limit: int = 100) -> List[Dict]:
        """Process batch of active signals."""
        # Get active signals
        signals = self._get_active_signals(limit)

        if not signals:
            logger.info("No active signals to process")
            return []

        logger.info(f"Processing {len(signals)} signals through multi-stage filter")

        results = []
        passed_counts = {'stage1': 0, 'stage3': 0, 'stage4': 0, 'final': 0}

        # Process signals in parallel batches
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.process_signal, signal): signal for signal in signals}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    results.append(result)

                    # Track pass rates
                    if result['stage1_passed']:
                        passed_counts['stage1'] += 1
                    if result['stage3_passed']:
                        passed_counts['stage3'] += 1
                    if result['stage4_passed']:
                        passed_counts['stage4'] += 1
                    if result['final_decision']:
                        passed_counts['final'] += 1

                except Exception as e:
                    logger.error(f"Error processing signal: {e}")

        # Log statistics
        total = len(signals)
        logger.info(f"\n📊 MULTI-STAGE FILTERING RESULTS:")
        logger.info(f"  Total signals: {total}")
        logger.info(f"  Stage 1 passed: {passed_counts['stage1']} ({passed_counts['stage1'] / total * 100:.1f}%)")
        logger.info(f"  Stage 3 passed: {passed_counts['stage3']} ({passed_counts['stage3'] / total * 100:.1f}%)")
        logger.info(f"  Stage 4 passed: {passed_counts['stage4']} ({passed_counts['stage4'] / total * 100:.1f}%)")
        logger.info(f"  ✅ Final passed: {passed_counts['final']} ({passed_counts['final'] / total * 100:.1f}%)")

        # Save results to database
        self._save_results(results)

        return results

    def _get_active_signals(self, limit: int) -> List[Dict]:
        """Get active signals from database."""
        # Use the same query as original production_predictor
        query = """
        WITH active_signals AS (
            SELECT
                sh.id,
                sh.timestamp,
                sh.trading_pair_id,
                sh.pair_symbol,
                sh.total_score,
                CASE
                    WHEN sh.total_score > 0 THEN 'BUY'
                    ELSE 'SELL'
                END AS signal_type,
                ind.rsi,
                ind.volume_zscore,
                ind.price_change_pct,
                ind.buy_ratio,
                ind.cvd_delta
            FROM fas.scoring_history sh
            LEFT JOIN LATERAL (
                SELECT *
                FROM fas.indicators i
                WHERE i.trading_pair_id = sh.trading_pair_id
                  AND i.timeframe = '15m'::fas.timeframe_enum
                  AND i.timestamp <= sh.timestamp
                ORDER BY i.timestamp DESC
                LIMIT 1
            ) ind ON true
            WHERE sh.is_active = true
                AND NOT public.is_stablecoin_pair(sh.trading_pair_id)
            LIMIT %s
        )
        SELECT * FROM active_signals
        """

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return cur.fetchall()

    def _save_results(self, results: List[Dict]):
        """Save prediction results to database."""
        if not results:
            return

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                for result in results:
                    try:
                        cur.execute("""
                            INSERT INTO ml.multistage_predictions (
                                signal_id, signal_timestamp, trading_pair_id, pair_symbol, signal_type,
                                stage1_model_id, stage1_probability, stage1_passed, stage1_time_ms,
                                detected_regime, regime_confidence,
                                stage3_model_id, stage3_probability, stage3_threshold, stage3_passed, stage3_time_ms,
                                stage4_model_id, stage4_probability, stage4_dynamic_threshold, stage4_passed, stage4_time_ms,
                                final_decision, final_confidence, total_processing_time_ms
                            ) VALUES (
                                %(signal_id)s, %(signal_timestamp)s, %(trading_pair_id)s, %(pair_symbol)s, %(signal_type)s,
                                %(stage1_model_id)s, %(stage1_probability)s, %(stage1_passed)s, %(stage1_time_ms)s,
                                %(detected_regime)s, %(regime_confidence)s,
                                %(stage3_model_id)s, %(stage3_probability)s, %(stage3_threshold)s, %(stage3_passed)s, %(stage3_time_ms)s,
                                %(stage4_model_id)s, %(stage4_probability)s, %(stage4_dynamic_threshold)s, %(stage4_passed)s, %(stage4_time_ms)s,
                                %(final_decision)s, %(final_confidence)s, %(total_processing_time_ms)s
                            )
                            ON CONFLICT (signal_id) DO UPDATE SET
                                final_decision = EXCLUDED.final_decision,
                                final_confidence = EXCLUDED.final_confidence,
                                predicted_at = NOW()
                        """, result)

                        # Mark signal as processed
                        cur.execute(
                            "UPDATE fas.scoring_history SET is_active = false WHERE id = %s",
                            (result['signal_id'],)
                        )

                    except Exception as e:
                        logger.error(f"Failed to save result for signal {result['signal_id']}: {e}")
                        conn.rollback()
                        continue

                conn.commit()
                logger.info(f"💾 Saved {len(results)} predictions to database")


def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("MULTI-STAGE PREDICTION SYSTEM")
    logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    predictor = MultiStagePredictor()
    results = predictor.process_batch(limit=500)

    # Show signals to trade
    trades = [r for r in results if r['final_decision']]
    if trades:
        logger.info(f"\n🎯 {len(trades)} SIGNALS TO TRADE:")
        for trade in sorted(trades, key=lambda x: x['final_confidence'], reverse=True)[:10]:
            logger.info(f"  {trade['signal_type']} {trade['pair_symbol']}: "
                        f"confidence={trade['final_confidence']:.3f}")

    logger.info("\n✅ Processing complete!")


if __name__ == "__main__":
    main()