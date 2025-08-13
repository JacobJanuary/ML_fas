"""
Analyze why training might be too fast
"""

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()


def measure_data_loading():
    """Measure data loading time and volume."""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    print("="*60)
    print("DATA LOADING PERFORMANCE TEST")
    print("="*60)

    # Test 1: Load BUY data as in training script
    print("\n📊 Loading BUY signals...")
    start_time = time.time()

    query_buy = """
    SELECT *
    FROM fas.ml_training_data_v2
    WHERE signal_type = 'BUY'
        AND target IS NOT NULL
        AND timestamp < NOW() - INTERVAL '12 hours'
    ORDER BY timestamp
    """

    df_buy = pd.read_sql(query_buy, conn)
    buy_load_time = time.time() - start_time

    print(f"  Loaded: {len(df_buy):,} records")
    print(f"  Time: {buy_load_time:.2f} seconds")
    print(f"  Speed: {len(df_buy)/buy_load_time:.0f} records/second")
    print(f"  Memory: {df_buy.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Test 2: Load SELL data
    print("\n📊 Loading SELL signals...")
    start_time = time.time()

    query_sell = """
    SELECT *
    FROM fas.ml_training_data_v2
    WHERE signal_type = 'SELL'
        AND target IS NOT NULL
        AND timestamp < NOW() - INTERVAL '12 hours'
    ORDER BY timestamp
    """

    df_sell = pd.read_sql(query_sell, conn)
    sell_load_time = time.time() - start_time

    print(f"  Loaded: {len(df_sell):,} records")
    print(f"  Time: {sell_load_time:.2f} seconds")
    print(f"  Speed: {len(df_sell)/sell_load_time:.0f} records/second")
    print(f"  Memory: {df_sell.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    conn.close()

    # Test 3: Quick model training simulation
    print("\n🤖 Simulating model training...")

    if len(df_buy) > 1000:
        # Prepare features (simplified)
        exclude_cols = [
            'target', 'timestamp', 'id', 'pair_symbol', 'trading_pair_id',
            'signal_type', 'signal_strength', 'market_regime'
        ]

        # Exclude meta fields
        meta_cols = [col for col in df_buy.columns if col.startswith('_meta_')]
        exclude_cols.extend(meta_cols)

        # Get numeric features only
        feature_cols = [col for col in df_buy.columns
                       if col not in exclude_cols
                       and df_buy[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        print(f"  Features: {len(feature_cols)}")

        # Handle NaN
        X = df_buy[feature_cols].fillna(0)
        y = df_buy['target'].astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print(f"  Train size: {len(X_train):,}")
        print(f"  Test size: {len(X_test):,}")

        # Train small model
        print("\n  Training RandomForest (100 trees, depth 5)...")
        start_time = time.time()

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f} seconds")

        # Evaluate
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        print(f"  Train accuracy: {train_score:.3f}")
        print(f"  Test accuracy: {test_score:.3f}")

        # Estimate full training time
        # XGBoost/LightGBM with optimization typically 10-20x slower
        estimated_full_time = train_time * 15 * 3  # 15x for complexity, 3 models
        print(f"\n  Estimated full training time: {estimated_full_time/60:.1f} minutes")

        if estimated_full_time < 60:  # Less than 1 minute
            print("  ⚠️ WARNING: Training seems too fast!")
            print("     Possible issues:")
            print("     - Not enough hyperparameter trials")
            print("     - Models are too simple")
            print("     - Early stopping too aggressive")
        else:
            print("  ✅ Training time seems reasonable")

    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    total_data = len(df_buy) + len(df_sell)
    total_load_time = buy_load_time + sell_load_time

    print(f"Total data: {total_data:,} records")
    print(f"Total load time: {total_load_time:.2f} seconds")
    print(f"Features: ~{len(feature_cols)} columns")

    if total_data < 50000:
        print("\n❌ INSUFFICIENT DATA!")
        print("   This explains fast training")
    elif total_data < 100000:
        print("\n⚠️ DATA VOLUME IS BORDERLINE")
        print("   Training might be faster than expected")
    else:
        print("\n✅ DATA VOLUME IS GOOD")
        print("   Training should take 10-30 minutes with proper optimization")

    # Check for data quality issues
    print("\n📋 Data Quality Checks:")

    # Check for constant columns
    constant_cols = []
    for col in feature_cols[:20]:  # Check first 20 features
        if X[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"  ⚠️ Found {len(constant_cols)} constant columns (no variance)")

    # Check for high correlation with target (potential leakage)
    high_corr_cols = []
    for col in feature_cols[:20]:
        corr = X[col].corr(y)
        if abs(corr) > 0.5:
            high_corr_cols.append((col, corr))

    if high_corr_cols:
        print(f"  ⚠️ Found {len(high_corr_cols)} columns with high correlation to target:")
        for col, corr in high_corr_cols[:5]:
            print(f"     {col}: {corr:.3f}")

    # Check class balance
    class_balance = y.value_counts(normalize=True)
    print(f"\n  Class balance:")
    print(f"    Class 0: {class_balance.get(0, 0):.1%}")
    print(f"    Class 1: {class_balance.get(1, 0):.1%}")

    if abs(class_balance.get(0, 0) - 0.5) > 0.3:
        print("    ⚠️ Classes are imbalanced")


def check_training_config():
    """Check training configuration that might affect speed."""

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION CHECK")
    print("="*60)

    print("\nCommon issues that cause fast training:")
    print("1. ❓ n_trials too low (should be 30-50 for optimization)")
    print("2. ❓ early_stopping_rounds too aggressive (should be 30-50)")
    print("3. ❓ max_depth too shallow (should be 4-6)")
    print("4. ❓ n_estimators too low (should be 100-300)")
    print("5. ❓ Using --quick mode (reduces trials to 5)")

    print("\nRecommended settings for proper training:")
    print("  - n_trials: 30 (minimum)")
    print("  - n_estimators: 100-300")
    print("  - max_depth: 4-6")
    print("  - early_stopping_rounds: 30")
    print("  - Don't use --quick for production")

    print("\nTo check your settings:")
    print("  Look at ml_training_v2.log for actual parameters used")


if __name__ == "__main__":
    measure_data_loading()
    check_training_config()

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Run: python check_market_regime_function.py")
    print("   To understand market regime function status")
    print("\n2. If training is too fast, check ml_training_v2.log")
    print("   Look for n_trials and optimization settings")
    print("\n3. For production training, use:")
    print("   python ml_trading_model_training.py  (without --quick)")