"""
Apply aggressive filtering to existing models to achieve target metrics
"""

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import numpy as np
import joblib
import logging
from sklearn.metrics import precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def load_test_data(signal_type='BUY'):
    """
    Load recent test data
    """
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    # Load last 3 days of data for testing
    query = f"""
    SELECT *
    FROM fas.ml_training_data_v2
    WHERE signal_type = '{signal_type}'
        AND target IS NOT NULL
        AND timestamp >= NOW() - INTERVAL '5 days'
        AND timestamp < NOW() - INTERVAL '2 days'
    ORDER BY timestamp
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} {signal_type} signals for testing")
    logger.info(f"Base win rate: {df['target'].mean():.1%}")

    return df


def apply_multi_level_filter(df, predictions, signal_type):
    """
    Apply multi-level filtering strategy
    """
    # Level 1: ML probability threshold
    level1_thresholds = {
        'BUY': 0.65,  # High confidence for BUY
        'SELL': 0.60  # Slightly lower for SELL
    }

    # Level 2: Market regime filter
    good_regimes = {
        'BUY': ['BULL'],
        'SELL': ['BEAR', 'NEUTRAL']  # SELL works in BEAR and NEUTRAL
    }

    # Level 3: Signal strength filter
    min_strength = {
        'BUY': ['STRONG', 'VERY_STRONG'],
        'SELL': ['MODERATE', 'STRONG', 'VERY_STRONG']
    }

    # Level 4: Technical filters
    tech_filters = {
        'BUY': {
            'rsi_max': 70,  # Not overbought
            'volume_zscore_min': 0.5,  # Some volume activity
        },
        'SELL': {
            'rsi_min': 30,  # Not oversold
            'volume_zscore_max': -0.5,  # Selling pressure
        }
    }

    # Apply filters
    filter_mask = np.ones(len(df), dtype=bool)

    # Level 1: ML Probability
    threshold = level1_thresholds[signal_type]
    level1_mask = predictions >= threshold
    filter_mask &= level1_mask
    logger.info(f"Level 1 (ML prob >= {threshold}): {level1_mask.sum()} / {len(df)} signals pass")

    # Level 2: Market Regime
    level2_mask = df['market_regime'].isin(good_regimes[signal_type])
    filter_mask &= level2_mask
    logger.info(f"Level 2 (Market regime): {(filter_mask).sum()} / {len(df)} signals pass")

    # Level 3: Signal Strength
    level3_mask = df['signal_strength'].isin(min_strength[signal_type])
    filter_mask &= level3_mask
    logger.info(f"Level 3 (Signal strength): {(filter_mask).sum()} / {len(df)} signals pass")

    # Level 4: Technical Filters
    if signal_type == 'BUY':
        level4_mask = (
                (df['rsi'] <= tech_filters['BUY']['rsi_max']) &
                (df['volume_zscore'] >= tech_filters['BUY']['volume_zscore_min'])
        )
    else:
        level4_mask = (
                (df['rsi'] >= tech_filters['SELL']['rsi_min']) |
                (df['volume_zscore'] <= tech_filters['SELL']['volume_zscore_max'])
        )

    filter_mask &= level4_mask
    logger.info(f"Level 4 (Technical): {(filter_mask).sum()} / {len(df)} signals pass")

    # Additional filters based on patterns
    if 'pattern_count' in df.columns:
        # Prefer signals with 1-2 patterns (not too many)
        pattern_mask = df['pattern_count'].between(1, 2)
        filter_mask &= pattern_mask
        logger.info(f"Level 5 (Pattern count): {(filter_mask).sum()} / {len(df)} signals pass")

    return filter_mask


def evaluate_filtering_strategy(signal_type='BUY'):
    """
    Evaluate aggressive filtering on existing models
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Evaluating {signal_type} with Aggressive Filtering")
    logger.info(f"{'=' * 50}")

    # Load saved model
    model_path = f'models/{signal_type.lower()}_best_model.pkl'

    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data['model']
            original_threshold = model_data.get('threshold', 0.5)
        else:
            model = model_data
            original_threshold = 0.5
    except:
        logger.error(f"Could not load model from {model_path}")
        return None

    # Load preprocessor
    preprocessor = joblib.load(f'models/{signal_type.lower()}_preprocessor.pkl')

    # Load test data
    df = load_test_data(signal_type)

    # Prepare features
    X, y = preprocessor.prepare_features(df, is_training=False)

    # Get predictions
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except:
        logger.error("Error getting predictions")
        return None

    # Test different filtering strategies
    strategies = [
        {
            'name': 'Original Threshold',
            'filter': y_pred_proba >= original_threshold
        },
        {
            'name': 'High Threshold (0.7)',
            'filter': y_pred_proba >= 0.7
        },
        {
            'name': 'Very High Threshold (0.8)',
            'filter': y_pred_proba >= 0.8
        },
        {
            'name': 'Multi-Level Filter',
            'filter': apply_multi_level_filter(df, y_pred_proba, signal_type)
        }
    ]

    # Add combined strategies
    high_prob_mask = y_pred_proba >= 0.65
    right_regime_mask = (
            ((signal_type == 'BUY') & (df['market_regime'] == 'BULL')) |
            ((signal_type == 'SELL') & (df['market_regime'].isin(['BEAR', 'NEUTRAL'])))
    )
    strong_signal_mask = df['signal_strength'].isin(['STRONG', 'VERY_STRONG'])

    strategies.append({
        'name': 'High Prob + Right Regime',
        'filter': high_prob_mask & right_regime_mask
    })

    strategies.append({
        'name': 'High Prob + Strong Signal',
        'filter': high_prob_mask & strong_signal_mask
    })

    strategies.append({
        'name': 'All Three Conditions',
        'filter': high_prob_mask & right_regime_mask & strong_signal_mask
    })

    # Evaluate each strategy
    results = []

    for strategy in strategies:
        filter_mask = strategy['filter']
        num_trades = filter_mask.sum()

        if num_trades == 0:
            logger.info(f"\n{strategy['name']}: No trades pass filter")
            continue

        # Calculate metrics on filtered trades
        y_filtered = y[filter_mask]
        win_rate = y_filtered.mean()
        trades_pct = num_trades / len(df)

        # Expected profit
        expected_profit = win_rate * 3 - (1 - win_rate) * 3

        results.append({
            'strategy': strategy['name'],
            'num_trades': num_trades,
            'trades_pct': trades_pct,
            'win_rate': win_rate,
            'expected_profit': expected_profit
        })

        logger.info(f"\n{strategy['name']}:")
        logger.info(f"  Trades: {num_trades} ({trades_pct:.1%} of all)")
        logger.info(f"  Win rate: {win_rate:.1%}")
        logger.info(f"  Expected profit: {expected_profit:.1%}")

        if win_rate >= 0.60 and trades_pct <= 0.35 and trades_pct >= 0.10:
            logger.info(f"  ✅ MEETS TARGET CRITERIA!")

    return pd.DataFrame(results)


def find_optimal_combination():
    """
    Find the optimal filtering combination for both signal types
    """
    logger.info("=" * 60)
    logger.info("FINDING OPTIMAL FILTERING STRATEGY")
    logger.info("=" * 60)

    all_results = {}

    for signal_type in ['BUY', 'SELL']:
        results_df = evaluate_filtering_strategy(signal_type)
        if results_df is not None:
            all_results[signal_type] = results_df

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL STRATEGIES SUMMARY")
    logger.info("=" * 60)

    for signal_type, results in all_results.items():
        logger.info(f"\n{signal_type} Best Strategies:")
        logger.info("-" * 40)

        # Filter for strategies that meet criteria
        good_strategies = results[
            (results['win_rate'] >= 0.55) &
            (results['trades_pct'] >= 0.05) &
            (results['trades_pct'] <= 0.40)
            ].sort_values('expected_profit', ascending=False)

        if len(good_strategies) > 0:
            for _, row in good_strategies.head(3).iterrows():
                logger.info(f"{row['strategy']}:")
                logger.info(f"  Win rate: {row['win_rate']:.1%}")
                logger.info(f"  Trades: {row['trades_pct']:.1%}")
                logger.info(f"  Expected profit: {row['expected_profit']:.1%}")
        else:
            logger.info("No strategies meet the criteria")

    return all_results


if __name__ == "__main__":
    find_optimal_combination()