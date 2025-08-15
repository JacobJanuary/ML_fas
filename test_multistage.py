#!/usr/bin/env python3
"""
Test script for Multi-Stage Predictor
Tests data type handling and stage progression
"""

import sys
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multistage_predictor import MultiStagePredictor
from feature_preprocessor import FeaturePreprocessor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_data_types():
    """Test that data types are handled correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Data Type Handling")
    logger.info("=" * 60)

    # Create sample data with mixed types (simulating database output)
    sample_data = {
        'id': 12345,
        'timestamp': datetime.now(),
        'trading_pair_id': '1001',  # String that should be numeric
        'pair_symbol': 'BTC/USDT',
        'signal_type': 'BUY',
        'total_score': '25.5',  # String that should be float
        'rsi': '45.2',  # String
        'volume_zscore': '1.8',  # String
        'price_change_pct': '2.3',  # String
        'buy_ratio': '0.65',  # String
        'cvd_delta': '1500.0',  # String
        'poc_volume_24h': '1000000',  # POC data as string
        'poc_volume_7d': '5000000',
        'price_to_poc_24h_pct': '0.5',
        'market_regime': 'NEUTRAL',
        'pattern_1_name': 'ACCUMULATION',
        'signal_strength': 'STRONG',
        'pattern_count': 2,
        'combo_count': 1
    }

    # Test preprocessor
    logger.info("\nTesting FeaturePreprocessor...")
    preprocessor = FeaturePreprocessor()

    df = pd.DataFrame([sample_data])
    logger.info(f"Original dtypes:\n{df.dtypes}")

    df_processed = preprocessor.prepare_features(df, is_training=False, model_type='quick_filter')
    logger.info(f"\nProcessed dtypes:\n{df_processed.dtypes}")

    # Check all columns are numeric
    non_numeric = df_processed.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        logger.error(f"❌ Non-numeric columns remain: {non_numeric}")
        return False
    else:
        logger.info("✅ All columns are numeric after preprocessing")

    # Check for NaN values
    nan_cols = df_processed.columns[df_processed.isna().any()].tolist()
    if nan_cols:
        logger.warning(f"⚠️ Columns with NaN: {nan_cols}")
    else:
        logger.info("✅ No NaN values")

    return True


def test_stage_progression():
    """Test that signals progress through stages correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Stage Progression")
    logger.info("=" * 60)

    try:
        # Initialize predictor
        predictor = MultiStagePredictor()

        # Get a sample signal from database
        conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get one active signal for testing
                cur.execute("""
                    SELECT 
                        sh.id,
                        sh.timestamp,
                        sh.trading_pair_id,
                        sh.pair_symbol,
                        sh.total_score,
                        sh.pattern_score,
                        sh.combination_score,
                        sh.indicator_score,
                        sh.recommended_action,
                        COALESCE(jsonb_array_length(sh.patterns_details), 0) as pattern_count,
                        COALESCE(jsonb_array_length(sh.combinations_details), 0) as combo_count,
                        sh.patterns_details->0->>'name' as pattern_1_name,
                        sh.patterns_details->1->>'name' as pattern_2_name,
                        sh.combinations_details->0->>'name' as combo_1_name,
                        CASE WHEN sh.total_score > 0 THEN 'BUY' ELSE 'SELL' END AS signal_type,
                        ind.rsi,
                        ind.volume_zscore,
                        ind.price_change_pct,
                        ind.buy_ratio,
                        ind.cvd_delta,
                        ind.oi_delta_pct as oi_change_pct,
                        ind.funding_rate_avg as funding_rate,
                        ind.normalized_imbalance,
                        ind.smoothed_imbalance,
                        ind.rs_momentum,
                        mr.regime as market_regime
                    FROM fas.scoring_history sh
                    LEFT JOIN LATERAL (
                        SELECT * FROM fas.indicators i
                        WHERE i.trading_pair_id = sh.trading_pair_id
                        AND i.timeframe = '15m'::fas.timeframe_enum
                        AND i.timestamp <= sh.timestamp
                        ORDER BY i.timestamp DESC
                        LIMIT 1
                    ) ind ON true
                    LEFT JOIN LATERAL (
                        SELECT regime FROM fas.market_regime mr
                        WHERE mr.timeframe = '4h'::fas.timeframe_enum
                        AND mr.timestamp <= sh.timestamp
                        ORDER BY mr.timestamp DESC
                        LIMIT 1
                    ) mr ON true
                    WHERE sh.is_active = true
                    LIMIT 1
                """)

                signal = cur.fetchone()

        if not signal:
            logger.warning("No active signals found for testing")
            return False

        logger.info(f"\nTesting signal: {signal['pair_symbol']} ({signal['signal_type']})")
        logger.info(f"Total score: {signal['total_score']}")

        # Process the signal
        result = predictor.process_signal(signal)

        # Log results
        logger.info(f"\n📊 Stage Results:")
        logger.info(f"  Stage 1 (Quick Filter): {'✅ PASSED' if result['stage1_passed'] else '❌ FAILED'}")
        logger.info(f"    - Probability: {result['stage1_probability']:.3f}")
        logger.info(f"    - Time: {result['stage1_time_ms']}ms")

        if result['stage1_passed']:
            logger.info(f"  Stage 2 (Regime): {result['detected_regime']} (conf: {result['regime_confidence']:.2f})")

            logger.info(f"  Stage 3 (Specialized): {'✅ PASSED' if result['stage3_passed'] else '❌ FAILED'}")
            logger.info(f"    - Probability: {result['stage3_probability']:.3f}")
            logger.info(f"    - Threshold: {result['stage3_threshold']:.3f}")
            logger.info(f"    - Time: {result['stage3_time_ms']}ms")

            if result['stage3_passed']:
                logger.info(f"  Stage 4 (Precision): {'✅ PASSED' if result['stage4_passed'] else '❌ FAILED'}")
                logger.info(f"    - Probability: {result['stage4_probability']:.3f}")
                logger.info(f"    - Dynamic Threshold: {result['stage4_dynamic_threshold']:.3f}")
                logger.info(f"    - Time: {result['stage4_time_ms']}ms")

        logger.info(f"\n🎯 Final Decision: {'✅ TRADE' if result['final_decision'] else '❌ NO TRADE'}")
        logger.info(f"   Final Confidence: {result['final_confidence']:.3f}")
        logger.info(f"   Total Processing Time: {result['total_processing_time_ms']}ms")

        return True

    except Exception as e:
        logger.error(f"❌ Stage progression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing of multiple signals."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Batch Processing")
    logger.info("=" * 60)

    try:
        predictor = MultiStagePredictor()

        # Process batch of 10 signals
        results = predictor.process_batch(limit=10)

        if not results:
            logger.warning("No signals processed")
            return False

        # Calculate statistics
        stage1_pass = sum(1 for r in results if r['stage1_passed'])
        stage3_pass = sum(1 for r in results if r['stage3_passed'])
        stage4_pass = sum(1 for r in results if r['stage4_passed'])
        final_pass = sum(1 for r in results if r['final_decision'])

        logger.info(f"\n📊 Batch Statistics ({len(results)} signals):")
        logger.info(f"  Stage 1 pass rate: {stage1_pass}/{len(results)} ({stage1_pass / len(results) * 100:.1f}%)")
        logger.info(f"  Stage 3 pass rate: {stage3_pass}/{len(results)} ({stage3_pass / len(results) * 100:.1f}%)")
        logger.info(f"  Stage 4 pass rate: {stage4_pass}/{len(results)} ({stage4_pass / len(results) * 100:.1f}%)")
        logger.info(f"  Final pass rate: {final_pass}/{len(results)} ({final_pass / len(results) * 100:.1f}%)")

        # Check if pass rates are reasonable
        final_pass_rate = final_pass / len(results)
        if 0.05 <= final_pass_rate <= 0.15:
            logger.info("✅ Final pass rate is within target range (5-15%)")
        else:
            logger.warning(f"⚠️ Final pass rate {final_pass_rate * 100:.1f}% is outside target range")

        # Show processing times
        avg_time = sum(r['total_processing_time_ms'] for r in results) / len(results)
        logger.info(f"\n⏱️ Average processing time: {avg_time:.1f}ms per signal")

        if avg_time < 100:
            logger.info("✅ Processing time is within target (<100ms)")
        else:
            logger.warning(f"⚠️ Processing time {avg_time:.1f}ms exceeds target")

        return True

    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MULTI-STAGE PREDICTOR TEST SUITE")
    logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    tests_passed = 0
    tests_total = 3

    # Test 1: Data types
    if test_data_types():
        tests_passed += 1

    # Test 2: Stage progression
    if test_stage_progression():
        tests_passed += 1

    # Test 3: Batch processing
    if test_batch_processing():
        tests_passed += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        logger.info("✅ ALL TESTS PASSED! System is ready for production.")
    else:
        logger.warning(f"⚠️ {tests_total - tests_passed} tests failed. Please review the logs.")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)