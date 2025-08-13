"""
Diagnose training data availability and market regime
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def check_data_availability():
    """Check how much data is available for training."""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("=" * 60)
    print("TRAINING DATA AVAILABILITY CHECK")
    print("=" * 60)

    # 1. Check total data in ml_training_data_v2
    cur.execute("""
        SELECT 
            COUNT(*) as total_count,
            MIN(timestamp) as min_date,
            MAX(timestamp) as max_date,
            COUNT(DISTINCT DATE(timestamp)) as days_count
        FROM fas.ml_training_data_v2
        WHERE target IS NOT NULL
    """)

    result = cur.fetchone()
    print(f"\n📊 Total data in ml_training_data_v2:")
    print(f"  Records: {result['total_count']:,}")
    print(f"  Date range: {result['min_date']} to {result['max_date']}")
    print(f"  Days covered: {result['days_count']}")

    # 2. Check data available for training (with time filter)
    for hours_offset in [12, 24, 48, 72]:
        cur.execute(f"""
            SELECT 
                signal_type,
                COUNT(*) as count,
                AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate
            FROM fas.ml_training_data_v2
            WHERE target IS NOT NULL
                AND timestamp < NOW() - INTERVAL '{hours_offset} hours'
            GROUP BY signal_type
            ORDER BY signal_type
        """)

        print(f"\n⏰ Data older than {hours_offset} hours:")
        for row in cur.fetchall():
            print(f"  {row['signal_type']}: {row['count']:,} signals ({row['win_rate'] * 100:.1f}% win rate)")

    # 3. Check market regime distribution
    cur.execute("""
        SELECT 
            market_regime,
            signal_type,
            COUNT(*) as count,
            AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate
        FROM fas.ml_training_data_v2
        WHERE target IS NOT NULL
            AND timestamp < NOW() - INTERVAL '12 hours'
        GROUP BY market_regime, signal_type
        ORDER BY market_regime, signal_type
    """)

    print(f"\n🎯 Market Regime Distribution (>12h old):")
    print(f"{'Regime':<10} {'Signal':<6} {'Count':>8} {'Win Rate':>10}")
    print("-" * 40)
    for row in cur.fetchall():
        print(f"{row['market_regime']:<10} {row['signal_type']:<6} {row['count']:>8,} {row['win_rate'] * 100:>9.1f}%")

    # 4. Test market regime function
    print(f"\n🔍 Testing fas.get_market_regime_at_time():")

    # Get some recent timestamps
    cur.execute("""
        SELECT DISTINCT timestamp
        FROM fas.ml_training_data_v2
        ORDER BY timestamp DESC
        LIMIT 5
    """)

    timestamps = [row['timestamp'] for row in cur.fetchall()]

    for ts in timestamps:
        # Get stored regime
        cur.execute("""
            SELECT market_regime
            FROM fas.ml_training_data_v2
            WHERE timestamp = %s
            LIMIT 1
        """, (ts,))
        stored_regime = cur.fetchone()['market_regime']

        # Get dynamic regime
        cur.execute("""
            SELECT fas.get_market_regime_at_time(%s, '15m'::fas.timeframe_enum) as regime
        """, (ts,))
        dynamic_regime = cur.fetchone()['regime']

        match = "✅" if stored_regime == dynamic_regime else "❌"
        print(f"  {ts}: Stored={stored_regime:<8} Dynamic={dynamic_regime:<8} {match}")

    # 5. Check if view needs refresh
    cur.execute("""
        SELECT 
            schemaname,
            matviewname,
            last_refresh
        FROM pg_stat_user_tables t
        JOIN pg_matviews m ON t.tablename = m.matviewname
        WHERE matviewname = 'ml_training_data_v2'
    """)

    result = cur.fetchone()
    if result:
        print(f"\n🔄 Materialized View Status:")
        print(f"  Last refresh: {result.get('last_refresh', 'Never')}")

        # Check how stale the data is
        cur.execute("""
            SELECT NOW() - MAX(timestamp) as data_age
            FROM fas.ml_training_data_v2
        """)
        age = cur.fetchone()['data_age']
        print(f"  Data age: {age}")

        if age and age.total_seconds() > 48 * 3600:
            print(f"  ⚠️ WARNING: Data is {age.days} days old! Need to refresh view")
            print(f"\n  Run: REFRESH MATERIALIZED VIEW fas.ml_training_data_v2;")

    # 6. Check for NULL values in critical columns
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN market_regime IS NULL THEN 1 ELSE 0 END) as null_regime,
            SUM(CASE WHEN close_price IS NULL THEN 1 ELSE 0 END) as null_price,
            SUM(CASE WHEN total_score IS NULL THEN 1 ELSE 0 END) as null_score,
            SUM(CASE WHEN price_to_poc_24h_pct IS NULL THEN 1 ELSE 0 END) as null_poc
        FROM fas.ml_training_data_v2
        WHERE target IS NOT NULL
            AND timestamp < NOW() - INTERVAL '12 hours'
    """)

    result = cur.fetchone()
    print(f"\n⚠️ NULL Values Check:")
    print(f"  Market regime: {result['null_regime']:,} ({result['null_regime'] / result['total'] * 100:.1f}%)")
    print(f"  Close price: {result['null_price']:,} ({result['null_price'] / result['total'] * 100:.1f}%)")
    print(f"  Total score: {result['null_score']:,} ({result['null_score'] / result['total'] * 100:.1f}%)")
    print(f"  POC 24h: {result['null_poc']:,} ({result['null_poc'] / result['total'] * 100:.1f}%)")

    conn.close()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if result['total_count'] < 100000:
        print("❌ Not enough data! Need at least 100,000 records")
        print("   1. Check if view is populated correctly")
        print("   2. Consider reducing time filter")
    else:
        print("✅ Sufficient data for training")

    print("\n💡 For prediction service:")
    print("   Always use fas.get_market_regime_at_time() for real-time regime")
    print("   Don't rely on stored market_regime from training data")


def test_market_regime_function():
    """Test the market regime function with different parameters."""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()

    print("\n" + "=" * 60)
    print("MARKET REGIME FUNCTION TEST")
    print("=" * 60)

    # Test with current time
    cur.execute("""
        SELECT 
            fas.get_market_regime_at_time(NOW(), '15m'::fas.timeframe_enum) as regime_15m,
            fas.get_market_regime_at_time(NOW(), '1h'::fas.timeframe_enum) as regime_1h,
            fas.get_market_regime_at_time(NOW(), '4h'::fas.timeframe_enum) as regime_4h
    """)

    result = cur.fetchone()
    print(f"\nCurrent market regime:")
    print(f"  15m: {result[0]}")
    print(f"  1h:  {result[1]}")
    print(f"  4h:  {result[2]}")

    # Test historical consistency
    print(f"\nHistorical regime changes (last 24h):")

    for hours_ago in range(0, 25, 4):
        cur.execute("""
            SELECT fas.get_market_regime_at_time(
                NOW() - INTERVAL '%s hours',
                '4h'::fas.timeframe_enum
            ) as regime
        """, (hours_ago,))

        regime = cur.fetchone()[0]
        print(f"  {hours_ago:2d}h ago: {regime}")

    conn.close()


def check_training_speed():
    """Analyze why training might be too fast."""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    print("\n" + "=" * 60)
    print("TRAINING SPEED ANALYSIS")
    print("=" * 60)

    # Simulate data loading as in training script
    query_buy = """
        SELECT COUNT(*) as count
        FROM fas.ml_training_data_v2
        WHERE signal_type = 'BUY'
            AND target IS NOT NULL
            AND timestamp < NOW() - INTERVAL '12 hours'
    """

    query_sell = """
        SELECT COUNT(*) as count
        FROM fas.ml_training_data_v2
        WHERE signal_type = 'SELL'
            AND target IS NOT NULL
            AND timestamp < NOW() - INTERVAL '12 hours'
    """

    cur = conn.cursor()

    cur.execute(query_buy)
    buy_count = cur.fetchone()[0]

    cur.execute(query_sell)
    sell_count = cur.fetchone()[0]

    print(f"\nData that would be loaded for training:")
    print(f"  BUY signals: {buy_count:,}")
    print(f"  SELL signals: {sell_count:,}")
    print(f"  Total: {buy_count + sell_count:,}")

    if buy_count < 10000 or sell_count < 10000:
        print("\n❌ TOO LITTLE DATA!")
        print("   This explains why training is so fast")
        print("\n   Possible causes:")
        print("   1. View not refreshed recently")
        print("   2. Time filter too restrictive (12 hours)")
        print("   3. Data pipeline issues")

        # Check without time filter
        cur.execute("""
            SELECT 
                signal_type,
                COUNT(*) as count
            FROM fas.ml_training_data_v2
            WHERE target IS NOT NULL
            GROUP BY signal_type
        """)

        print(f"\n   Without time filter:")
        for row in cur.fetchall():
            print(f"   {row[0]}: {row[1]:,} signals")
    else:
        expected_time = (buy_count + sell_count) / 1000  # rough estimate: 1 second per 1000 samples
        print(f"\n✅ Sufficient data")
        print(f"   Expected training time: {expected_time / 60:.1f} minutes")

    conn.close()


if __name__ == "__main__":
    check_data_availability()
    test_market_regime_function()
    check_training_speed()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. If data is insufficient, refresh the materialized view:")
    print("   REFRESH MATERIALIZED VIEW fas.ml_training_data_v2;")
    print("\n2. Consider reducing time filter in training script:")
    print("   Change 'INTERVAL 12 hours' to 'INTERVAL 6 hours' or remove it")
    print("\n3. Update prediction service to use dynamic market regime:")
    print("   fas.get_market_regime_at_time(timestamp, '4h'::fas.timeframe_enum)")