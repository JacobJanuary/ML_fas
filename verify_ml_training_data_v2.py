"""
Verify the new ML training data view
"""

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def verify_new_view():
    """
    Comprehensive verification of the new ML training data view
    """

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    cur = conn.cursor()

    print("=" * 60)
    print("VERIFYING ML TRAINING DATA V2")
    print("=" * 60)

    # 1. Check if view exists and has data
    cur.execute("""
        SELECT COUNT(*) 
        FROM fas.ml_training_data_v2
    """)
    total_count = cur.fetchone()[0]
    print(f"\n✅ Total records: {total_count:,}")

    if total_count == 0:
        print("❌ ERROR: No data in view!")
        return

    # 2. Check win rates by signal type
    cur.execute("""
        SELECT 
            signal_type,
            COUNT(*) as count,
            AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(CASE WHEN target THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN NOT target THEN 1 ELSE 0 END) as losses
        FROM fas.ml_training_data_v2
        GROUP BY signal_type
        ORDER BY signal_type
    """)

    print("\n📊 Win Rates by Signal Type:")
    print("-" * 40)
    for row in cur.fetchall():
        signal_type, count, win_rate, wins, losses = row
        print(f"{signal_type}: {win_rate * 100:.1f}% ({wins:,} wins / {losses:,} losses) from {count:,} signals")

    # 3. Check POC data coverage
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN price_to_poc_24h_pct IS NOT NULL THEN 1 ELSE 0 END) as has_poc_24h,
            SUM(CASE WHEN price_to_poc_7d_pct IS NOT NULL THEN 1 ELSE 0 END) as has_poc_7d,
            SUM(CASE WHEN price_to_poc_30d_pct IS NOT NULL THEN 1 ELSE 0 END) as has_poc_30d
        FROM fas.ml_training_data_v2
    """)

    total, has_24h, has_7d, has_30d = cur.fetchone()
    print(f"\n📍 POC Data Coverage:")
    print(f"  24h POC: {has_24h / total * 100:.1f}% ({has_24h:,}/{total:,})")
    print(f"  7d POC:  {has_7d / total * 100:.1f}% ({has_7d:,}/{total:,})")
    print(f"  30d POC: {has_30d / total * 100:.1f}% ({has_30d:,}/{total:,})")

    # 4. Check win rates by market regime
    cur.execute("""
        SELECT 
            signal_type,
            market_regime,
            COUNT(*) as count,
            AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate
        FROM fas.ml_training_data_v2
        GROUP BY signal_type, market_regime
        ORDER BY signal_type, market_regime
    """)

    print(f"\n🎯 Win Rates by Market Regime:")
    print("-" * 50)
    print(f"{'Signal':<6} {'Regime':<8} {'Count':>8} {'Win Rate':>10}")
    print("-" * 50)
    for row in cur.fetchall():
        signal_type, regime, count, win_rate = row
        indicator = "✅" if (
                (signal_type == 'BUY' and regime == 'BULL' and win_rate > 0.5) or
                (signal_type == 'SELL' and regime == 'BEAR' and win_rate > 0.5)
        ) else "❌"
        print(f"{signal_type:<6} {regime:<8} {count:>8,} {win_rate * 100:>9.1f}% {indicator}")

    # 5. Check pattern distribution
    cur.execute("""
        SELECT 
            pattern_count,
            COUNT(*) as signal_count,
            AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate
        FROM fas.ml_training_data_v2
        WHERE pattern_count IS NOT NULL
        GROUP BY pattern_count
        ORDER BY pattern_count
    """)

    print(f"\n📈 Pattern Count Distribution:")
    print("-" * 40)
    for row in cur.fetchall():
        pattern_count, signal_count, win_rate = row
        print(f"{pattern_count} patterns: {signal_count:,} signals ({win_rate * 100:.1f}% win rate)")

    # 6. Check for dangerous columns
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'fas' 
            AND table_name = 'ml_training_data_v2'
            AND column_name LIKE '%%_meta%%'
        ORDER BY column_name
    """)

    meta_cols = [row[0] for row in cur.fetchall()]
    if meta_cols:
        print(f"\n⚠️ Meta columns (DO NOT use for training):")
        for col in meta_cols:
            print(f"  - {col}")

    # 7. Sample data quality check
    cur.execute("""
        SELECT 
            signal_type,
            close_price,
            price_to_poc_24h_pct,
            rsi,
            volume_zscore,
            market_regime,
            signal_strength,
            target,
            _meta_time_to_outcome_hours
        FROM fas.ml_training_data_v2
        WHERE _meta_time_to_outcome_hours IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5
    """)

    print(f"\n🔍 Sample Records:")
    print("-" * 80)
    cols = ['Type', 'Price', 'POC%', 'RSI', 'Vol-Z', 'Regime', 'Strength', 'Win?', 'Hours']
    print('  '.join(f"{col:<10}" for col in cols))
    print("-" * 80)

    for row in cur.fetchall():
        signal_type, price, poc_pct, rsi, vol_z, regime, strength, target, hours = row
        print(f"{signal_type:<10}  "
              f"{price:<10.0f}  "
              f"{poc_pct or 0:<10.1f}  "
              f"{rsi:<10.1f}  "
              f"{vol_z:<10.1f}  "
              f"{regime:<10}  "
              f"{strength:<10}  "
              f"{'WIN' if target else 'LOSS':<10}  "
              f"{hours or 48:<10.1f}")

    # 8. Check temporal distribution
    cur.execute("""
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as count,
            AVG(CASE WHEN target THEN 1.0 ELSE 0.0 END) as win_rate
        FROM fas.ml_training_data_v2
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 10
    """)

    print(f"\n📅 Recent Days Distribution:")
    print("-" * 40)
    for row in cur.fetchall():
        date, count, win_rate = row
        print(f"{date}: {count:,} signals ({win_rate * 100:.1f}% win rate)")

    conn.close()

    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_new_view()