"""
Emergency fix for data loading issues in ML training
"""

import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def load_and_clean_data(signal_type='BUY'):
    """
    Load data with proper handling of NULLs and dangerous columns
    """

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    # Load ALL data first (since specific column selection doesn't work)
    query = f"""
    SELECT *
    FROM fas.ml_new_training_data
    WHERE signal_type = '{signal_type}'
        AND target_with_stoploss IS NOT NULL
        AND timestamp < NOW() - INTERVAL '12 hours'  -- Reduced from 60 to get more data
    ORDER BY timestamp
    """

    logger.info(f"Loading {signal_type} data...")
    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} rows")

    # CRITICAL: Remove dangerous columns IMMEDIATELY
    dangerous_cols = [
        'outcome_timestamp', 'outcome',
        'max_profit_pct', 'max_loss_pct', 'hours_to_outcome',
        'id', 'trading_pair_id', 'created_at'
    ]

    cols_to_drop = [col for col in dangerous_cols if col in df.columns]
    if cols_to_drop:
        logger.warning(f"❌ REMOVING DANGEROUS COLUMNS: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Handle POC columns - convert to relative values
    if 'close_price' in df.columns:
        # Create relative POC features instead of absolute values
        if 'poc_24h' in df.columns:
            df['price_to_poc_24h_pct'] = np.where(
                df['poc_24h'].notna() & (df['poc_24h'] != 0),
                (df['close_price'] - df['poc_24h']) / df['poc_24h'] * 100,
                0  # Default to 0 if POC is missing
            )

        if 'poc_7d' in df.columns:
            df['price_to_poc_7d_pct'] = np.where(
                df['poc_7d'].notna() & (df['poc_7d'] != 0),
                (df['close_price'] - df['poc_7d']) / df['poc_7d'] * 100,
                0
            )

        if 'poc_30d' in df.columns:
            df['price_to_poc_30d_pct'] = np.where(
                df['poc_30d'].notna() & (df['poc_30d'] != 0),
                (df['close_price'] - df['poc_30d']) / df['poc_30d'] * 100,
                0
            )

        # Drop absolute POC values - they're useless for ML
        poc_cols_to_drop = ['poc_24h', 'poc_7d', 'poc_30d', 'poc_calculated_at']
        df = df.drop(columns=[col for col in poc_cols_to_drop if col in df.columns])
        logger.info("✅ Converted POC to relative values")

    # Handle pattern columns intelligently
    # Create aggregated pattern features instead of handling 10 separate patterns
    pattern_cols = [col for col in df.columns if 'pattern_' in col and '_name' not in col]

    if pattern_cols:
        # Count number of patterns (non-null pattern names)
        pattern_name_cols = [col for col in df.columns if col.startswith('pattern_') and col.endswith('_name')]
        df['num_patterns'] = df[pattern_name_cols].notna().sum(axis=1)

        # Sum total impact and average confidence (only for non-null values)
        impact_cols = [col for col in pattern_cols if 'impact' in col]
        conf_cols = [col for col in pattern_cols if 'confidence' in col]

        if impact_cols:
            df['total_pattern_impact'] = df[impact_cols].sum(axis=1, skipna=True)
            df['max_pattern_impact'] = df[impact_cols].max(axis=1, skipna=True)

        if conf_cols:
            df['avg_pattern_confidence'] = df[conf_cols].mean(axis=1, skipna=True)
            df['max_pattern_confidence'] = df[conf_cols].max(axis=1, skipna=True)

        logger.info(f"✅ Created aggregated pattern features from {len(pattern_cols)} columns")

        # Keep only first 3 patterns details (most signals have 1-3 patterns)
        patterns_to_keep = []
        for i in range(1, 4):  # Keep patterns 1-3
            for suffix in ['name', 'impact', 'confidence']:
                col = f'pattern_{i}_{suffix}'
                if col in df.columns:
                    patterns_to_keep.append(col)
                    # Fill missing values appropriately
                    if suffix == 'name':
                        df[col] = df[col].fillna('none')
                    else:
                        df[col] = df[col].fillna(0)

        # Drop patterns 4-10
        patterns_to_drop = []
        for i in range(4, 11):
            for suffix in ['name', 'impact', 'confidence']:
                col = f'pattern_{i}_{suffix}'
                if col in df.columns:
                    patterns_to_drop.append(col)

        if patterns_to_drop:
            df = df.drop(columns=patterns_to_drop)
            logger.info(f"✅ Dropped {len(patterns_to_drop)} redundant pattern columns")

    # Handle combination columns similarly
    combo_cols = [col for col in df.columns if 'combo_' in col and '_name' not in col]

    if combo_cols:
        # Count number of combinations
        combo_name_cols = [col for col in df.columns if col.startswith('combo_') and col.endswith('_name')]
        df['num_combos'] = df[combo_name_cols].notna().sum(axis=1)

        # Aggregate scores and confidences
        score_cols = [col for col in combo_cols if 'score' in col]
        conf_cols = [col for col in combo_cols if 'confidence' in col]

        if score_cols:
            df['total_combo_score'] = df[score_cols].sum(axis=1, skipna=True)
            df['max_combo_score'] = df[score_cols].max(axis=1, skipna=True)

        if conf_cols:
            df['avg_combo_confidence'] = df[conf_cols].mean(axis=1, skipna=True)
            df['max_combo_confidence'] = df[conf_cols].max(axis=1, skipna=True)

        logger.info(f"✅ Created aggregated combo features from {len(combo_cols)} columns")

        # Keep only first 2 combos (most signals have 0-2)
        combos_to_keep = []
        for i in range(1, 3):  # Keep combos 1-2
            for suffix in ['name', 'score', 'confidence']:
                col = f'combo_{i}_{suffix}'
                if col in df.columns:
                    combos_to_keep.append(col)
                    # Fill missing values appropriately
                    if suffix == 'name':
                        df[col] = df[col].fillna('none')
                    else:
                        df[col] = df[col].fillna(0)

        # Drop combos 3-6
        combos_to_drop = []
        for i in range(3, 7):
            for suffix in ['name', 'score', 'confidence']:
                col = f'combo_{i}_{suffix}'
                if col in df.columns:
                    combos_to_drop.append(col)

        if combos_to_drop:
            df = df.drop(columns=combos_to_drop)
            logger.info(f"✅ Dropped {len(combos_to_drop)} redundant combo columns")

    # Handle other NULL values more intelligently
    # For numeric columns that aren't pattern/combo related
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0 and null_count < len(df) * 0.5:  # Less than 50% null
            # Use median for filling
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            if null_count > 100:
                logger.info(f"  Filled {null_count} NULLs in {col} with median {median_val:.2f}")

    # Final check for any remaining dangerous columns
    remaining_dangerous = [col for col in df.columns if any(
        word in col.lower() for word in ['outcome', 'result', 'profit', 'loss', 'hours_to', 'exit']
    )]
    if remaining_dangerous:
        logger.error(f"❌ STILL HAVE DANGEROUS COLUMNS: {remaining_dangerous}")
        df = df.drop(columns=remaining_dangerous)

    # Log final stats
    logger.info(f"\n📊 Final dataset stats:")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Win rate: {df['target_with_stoploss'].mean():.1%}")
    logger.info(f"  Market regime distribution:")
    logger.info(f"{df['market_regime'].value_counts()}")

    # Check temporal range
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def validate_data_quality(df):
    """
    Validate that data is clean and ready for training
    """
    issues = []

    # Check for dangerous columns
    dangerous_patterns = ['outcome', 'result', 'profit', 'loss', 'hours_to', 'exit', 'max_']
    dangerous_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in dangerous_patterns):
            dangerous_cols.append(col)

    if dangerous_cols:
        issues.append(f"Dangerous columns found: {dangerous_cols}")

    # Check win rate
    win_rate = df['target_with_stoploss'].mean()
    if win_rate > 0.7 or win_rate < 0.3:
        issues.append(f"Suspicious win rate: {win_rate:.1%}")

    # Check data volume
    if len(df) < 10000:
        issues.append(f"Insufficient data: only {len(df)} rows")

    # Check for excessive NULLs
    null_cols = df.isnull().sum()
    high_null_cols = null_cols[null_cols > len(df) * 0.5]
    if len(high_null_cols) > 0:
        issues.append(f"Columns with >50% NULLs: {high_null_cols.index.tolist()}")

    if issues:
        logger.error("❌ DATA QUALITY ISSUES:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ Data quality checks passed!")
        return True


if __name__ == "__main__":
    # Test with BUY signals
    logger.info("Testing data loading for BUY signals...")
    df_buy = load_and_clean_data('BUY')
    is_valid = validate_data_quality(df_buy)

    if is_valid:
        logger.info("\n✅ BUY data is ready for training!")
        logger.info(f"Feature columns ({len(df_buy.columns)}):")
        for col in sorted(df_buy.columns):
            if col not in ['timestamp', 'pair_symbol', 'target_with_stoploss']:
                null_pct = df_buy[col].isnull().sum() / len(df_buy) * 100
                logger.info(f"  - {col}: {null_pct:.1f}% null")

    # Test with SELL signals
    logger.info("\n" + "=" * 50)
    logger.info("Testing data loading for SELL signals...")
    df_sell = load_and_clean_data('SELL')
    is_valid = validate_data_quality(df_sell)

    if is_valid:
        logger.info("\n✅ SELL data is ready for training!")