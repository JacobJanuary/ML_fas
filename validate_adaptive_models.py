"""
Validation script for adaptive models.
Tests on data from 72-48 hours ago (signals with known outcomes).
Cannot test on last 48 hours as outcomes are not yet determined.
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_model(model_path, signal_type, test_start_hours=72, test_end_hours=48):
    """
    Validate saved adaptive model on recent data with known outcomes.

    Args:
        model_path: Path to saved model
        signal_type: 'BUY' or 'SELL'
        test_start_hours: Start of test window (hours ago)
        test_end_hours: End of test window (hours ago) - need time for outcome
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Validating {signal_type} model from {model_path}")
    logger.info(f"Test window: {test_start_hours}h to {test_end_hours}h ago")
    logger.info(f"{'='*60}")

    # Load model
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        threshold = model_data['threshold']
        feature_columns = model_data['feature_columns']
        window_days = model_data.get('window_days', 7)

        logger.info(f"Model loaded: {window_days}-day window, threshold={threshold:.3f}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

    # Connect to database
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    # Load test data (data with known outcomes, not used in training)
    query = f"""
    SELECT *
    FROM fas.ml_training_data_direct
    WHERE signal_type = '{signal_type}'
        AND target IS NOT NULL
        AND timestamp >= NOW() - INTERVAL '{test_start_hours} hours'
        AND timestamp < NOW() - INTERVAL '{test_end_hours} hours'
    ORDER BY timestamp
    """

    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    if len(df) == 0:
        logger.warning(f"No test data found for period {test_start_hours}-{test_end_hours} hours ago")
        return None

    logger.info(f"Loaded {len(df)} test records")
    base_wr = df['target'].mean()
    logger.info(f"Base win rate in test data: {base_wr:.1%}")

    # Prepare features (same as training)
    df_proc = df.copy()

    # Remove dangerous columns
    remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                  'signal_type', 'signal_strength']
    remove_cols += [col for col in df_proc.columns if col.startswith('_meta_')]

    for col in remove_cols:
        if col in df_proc.columns:
            df_proc = df_proc.drop(columns=[col])

    # Fix extreme values
    for col in ['poc_volume_7d', 'poc_volume_24h']:
        if col in df_proc.columns:
            q99 = df_proc[col].quantile(0.99)
            df_proc[col] = df_proc[col].clip(upper=q99)
            df_proc[f'{col}_log'] = np.log1p(df_proc[col])
            if df_proc[col].max() / (df_proc[col].quantile(0.95) + 1) > 100:
                df_proc = df_proc.drop(columns=[col])

    # Add time features
    df_proc['timestamp_hour'] = pd.to_datetime(df.index).hour
    df_proc['timestamp_day'] = pd.to_datetime(df.index).dayofweek

    # Add rolling features
    if len(df_proc) > 50:
        df_proc['recent_avg'] = df_proc['target'].rolling(window=50, min_periods=10).mean()
        df_proc['recent_std'] = df_proc['target'].rolling(window=50, min_periods=10).std()
        df_proc.fillna(method='bfill', inplace=True)
    else:
        # Use global average for small datasets
        df_proc['recent_avg'] = base_wr
        df_proc['recent_std'] = df_proc['target'].std()

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
        if col != 'target':
            df_proc[col] = pd.Categorical(df_proc[col].fillna('unknown')).codes

    # Select only features used in training
    missing_features = set(feature_columns) - set(df_proc.columns)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Add missing features as zeros
        for feat in missing_features:
            df_proc[feat] = 0

    X = df_proc[feature_columns].fillna(0)
    y = df['target'].astype(int)

    # Scale features
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

    # Make predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    trades_taken = y_pred.sum()
    trades_pct = trades_taken / len(y_pred)

    if trades_taken > 0:
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        win_rate = tp / trades_taken
        expected_profit = (tp * 3 - fp * 3) / trades_taken
    else:
        win_rate = 0
        expected_profit = 0

    logger.info(f"\n📊 VALIDATION RESULTS:")
    logger.info(f"Predictions range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    logger.info(f"Trades taken: {trades_taken}/{len(y_pred)} ({trades_pct:.1%})")
    logger.info(f"Win rate: {win_rate:.1%}")
    logger.info(f"Expected profit: {expected_profit:.1%}")

    # Check by market regime
    if 'market_regime' in df.columns:
        logger.info(f"\nBy Market Regime:")
        for regime in ['BULL', 'NEUTRAL', 'BEAR']:
            mask = df['market_regime'] == regime
            if mask.sum() > 0:
                regime_pred = y_pred[mask]
                regime_actual = y[mask]

                if regime_pred.sum() > 0:
                    regime_wr = ((regime_pred == 1) & (regime_actual == 1)).sum() / regime_pred.sum()
                    logger.info(f"  {regime}: {regime_pred.sum()}/{mask.sum()} trades, "
                              f"{regime_wr:.1%} win rate")

    # Performance assessment
    logger.info(f"\n🎯 TARGET ASSESSMENT:")
    if signal_type == 'BUY':
        target = 0.80
    else:
        target = 0.65

    if win_rate >= target and 0.15 <= trades_pct <= 0.35:
        logger.info(f"✅ TARGET ACHIEVED! ({target:.0%} target)")
    elif win_rate >= 0.60:
        logger.info(f"✅ ACCEPTABLE ({win_rate:.1%} >= 60%)")
    else:
        logger.info(f"❌ BELOW TARGET ({win_rate:.1%} < {target:.0%})")

    return {
        'signal_type': signal_type,
        'window_days': window_days,
        'test_period': f'{test_start_hours}-{test_end_hours}h ago',
        'test_samples': len(df),
        'trades_taken': trades_taken,
        'trades_pct': trades_pct,
        'win_rate': win_rate,
        'expected_profit': expected_profit
    }


def main():
    """Validate all saved adaptive models."""

    logger.info("="*60)
    logger.info("ADAPTIVE MODEL VALIDATION")
    logger.info("Testing on data from 72-48 hours ago (with known outcomes)")
    logger.info("="*60)

    results = []

    # Test different window models
    model_configs = [
        ('models/buy_adaptive_model.pkl', 'BUY'),
        ('models/sell_adaptive_model.pkl', 'SELL')
    ]

    for model_path, signal_type in model_configs:
        if os.path.exists(model_path):
            # Test on 72-48 hours ago window (outcomes are known)
            result = validate_model(model_path, signal_type, test_start_hours=72, test_end_hours=48)
            if result:
                results.append(result)
        else:
            logger.warning(f"Model not found: {model_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)

    for result in results:
        logger.info(f"\n{result['signal_type']} Model:")
        logger.info(f"  Test period: {result['test_period']}")
        logger.info(f"  Test samples: {result['test_samples']}")
        logger.info(f"  Trades taken: {result['trades_pct']:.1%}")
        logger.info(f"  Win rate: {result['win_rate']:.1%}")
        logger.info(f"  Expected profit: {result['expected_profit']:.1%}")

        if result['win_rate'] >= (0.80 if result['signal_type'] == 'BUY' else 0.65):
            logger.info(f"  🎯 MEETS TARGET!")

    logger.info("\n✅ Validation complete")
    logger.info("⚠️ Remember to retrain models daily due to market changes")


if __name__ == "__main__":
    main()