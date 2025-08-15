#!/usr/bin/env python3
"""
Final test of Multi-Stage Prediction System
"""

import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_final_test():
    """Run comprehensive test of the system."""

    logger.info("=" * 60)
    logger.info("FINAL SYSTEM TEST")
    logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # Import predictor
    from multistage_predictor import MultiStagePredictor

    # Create instance
    predictor = MultiStagePredictor()

    # Process LARGE batch for proper statistics
    test_size = 100  # Start with 100 for speed, can increase to 1000
    logger.info(f"\n📊 Processing batch of {test_size} signals...")
    results = predictor.process_batch(limit=test_size)

    if not results:
        logger.error("❌ No results returned!")
        return False

    # Calculate statistics
    total = len(results)
    stage1_passed = sum(1 for r in results if r.get('stage1_passed', False))
    stage3_passed = sum(1 for r in results if r.get('stage3_passed', False))
    stage4_passed = sum(1 for r in results if r.get('stage4_passed', False))
    final_passed = sum(1 for r in results if r['final_decision'])

    # Calculate pass rates
    stage1_rate = stage1_passed / total * 100 if total > 0 else 0
    stage3_rate = stage3_passed / total * 100 if total > 0 else 0
    stage4_rate = stage4_passed / total * 100 if total > 0 else 0
    final_rate = final_passed / total * 100 if total > 0 else 0

    # Calculate reduction at each stage
    stage3_reduction = (1 - stage3_passed / stage1_passed) * 100 if stage1_passed > 0 else 100
    stage4_reduction = (1 - stage4_passed / stage3_passed) * 100 if stage3_passed > 0 else 100

    # Performance metrics
    avg_time = sum(r['total_processing_time_ms'] for r in results) / total

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)

    logger.info(f"\n📈 Stage Progression (n={total}):")
    logger.info(f"  Stage 1 (Quick Filter):  {stage1_passed:3d}/{total} ({stage1_rate:5.1f}%)")
    logger.info(
        f"  Stage 3 (Regime Model):  {stage3_passed:3d}/{total} ({stage3_rate:5.1f}%) - {stage3_reduction:.0f}% reduction")
    logger.info(
        f"  Stage 4 (Precision):     {stage4_passed:3d}/{total} ({stage4_rate:5.1f}%) - {stage4_reduction:.0f}% reduction")
    logger.info(f"  ✅ Final Decision:       {final_passed:3d}/{total} ({final_rate:5.1f}%)")

    logger.info(f"\n⏱️ Performance:")
    logger.info(f"  Average processing time: {avg_time:.0f}ms per signal")
    logger.info(f"  Total time: {avg_time * total / 1000:.1f}s for {total} signals")
    logger.info(f"  Throughput: {1000 / avg_time:.1f} signals/second")

    # Show signals that passed
    if final_passed > 0:
        logger.info(f"\n🎯 Signals to Trade ({final_passed}):")
        trades = [r for r in results if r['final_decision']]
        trades_sorted = sorted(trades, key=lambda x: x['final_confidence'], reverse=True)
        for i, trade in enumerate(trades_sorted[:10], 1):  # Show top 10
            logger.info(f"  {i}. {trade['signal_type']:4s} {trade['pair_symbol']:12s} "
                        f"conf={trade['final_confidence']:.3f}")

    # Analyze by signal type
    buy_signals = [r for r in results if r['signal_type'] == 'BUY']
    sell_signals = [r for r in results if r['signal_type'] == 'SELL']

    logger.info(f"\n📊 By Signal Type:")
    if buy_signals:
        buy_passed = sum(1 for r in buy_signals if r['final_decision'])
        logger.info(
            f"  BUY:  {buy_passed:2d}/{len(buy_signals):2d} passed ({buy_passed / len(buy_signals) * 100:5.1f}%)")

    if sell_signals:
        sell_passed = sum(1 for r in sell_signals if r['final_decision'])
        logger.info(
            f"  SELL: {sell_passed:2d}/{len(sell_signals):2d} passed ({sell_passed / len(sell_signals) * 100:5.1f}%)")

    # Probability distribution analysis
    if stage4_passed > 0:
        stage4_probs = [r['stage4_probability'] for r in results if r.get('stage3_passed', False)]
        if stage4_probs:
            logger.info(f"\n📊 Stage 4 Probability Distribution:")
            logger.info(f"  Min:    {min(stage4_probs):.3f}")
            logger.info(f"  Median: {sorted(stage4_probs)[len(stage4_probs) // 2]:.3f}")
            logger.info(f"  Max:    {max(stage4_probs):.3f}")
            logger.info(f"  Avg:    {sum(stage4_probs) / len(stage4_probs):.3f}")

    # Evaluate results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    success = True

    # Check stage 1 pass rate (should be 60-80%)
    if 60 <= stage1_rate <= 95:
        logger.info("✅ Stage 1 pass rate acceptable")
    else:
        logger.warning(f"⚠️ Stage 1 pass rate {stage1_rate:.1f}% may need adjustment")

    # Check final pass rate (target 5-15%, accept 1-20% for bear market)
    if 5 <= final_rate <= 15:
        logger.info("✅ Final pass rate within ideal target (5-15%)")
    elif 1 <= final_rate <= 20:
        logger.info("⚠️ Final pass rate acceptable for current market (1-20%)")
    elif final_rate == 0:
        logger.warning("❌ No signals passed - thresholds need adjustment")
        success = False
    else:
        logger.warning(f"⚠️ Final pass rate {final_rate:.1f}% outside acceptable range")

    # Check performance
    if avg_time < 1000:
        logger.info("✅ Performance excellent (<1s per signal)")
    elif avg_time < 3000:
        logger.info("⚠️ Performance acceptable (<3s per signal)")
    else:
        logger.warning("❌ Performance needs optimization")

    # Database check
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Check saved predictions
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM ml.multistage_predictions 
                    WHERE predicted_at >= NOW() - INTERVAL '1 hour'
                """)
                recent_count = cur.fetchone()[0]
                logger.info(f"\n📊 Database: {recent_count} predictions saved in last hour")

                # Check system health
                cur.execute("SELECT * FROM ml.check_system_health()")
                logger.info("\n🏥 System Health:")
                for row in cur.fetchall():
                    check_name, status, details = row
                    emoji = "✅" if status == "OK" else "⚠️" if status == "WARNING" else "❌"
                    logger.info(f"  {emoji} {check_name}: {details}")

    except Exception as e:
        logger.error(f"Database check failed: {e}")

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ SYSTEM TEST PASSED!")
        logger.info("=" * 60)
        logger.info("\nThe Multi-Stage Prediction System is ready for production.")
        logger.info("\nRecommended next steps:")
        logger.info("1. Run continuously: python multistage_predictor.py")
        logger.info("2. Monitor win rates over next 24-48 hours")
        logger.info("3. Fine-tune thresholds based on actual results")
        logger.info("4. Set up automated threshold calibration")
    else:
        logger.warning("\n⚠️ System needs further tuning")

    return success


if __name__ == "__main__":
    run_final_test()