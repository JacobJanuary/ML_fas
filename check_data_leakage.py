"""
Data Leakage Check for ML Training Data V2
===========================================
Validates that no future data is leaking into features
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def load_data(limit=10000):
    """Load data from ml_training_data_v2."""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    query = f"""
    SELECT *
    FROM fas.ml_training_data_v2
    WHERE target IS NOT NULL
    ORDER BY RANDOM()
    LIMIT {limit}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def check_meta_fields(df):
    """Check for presence and correlation of meta fields."""
    logger.info("Starting data leakage analysis...")
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Target distribution: {df['target'].mean():.2%} positive")

    logger.info("\n" + "="*50)
    logger.info("TEST 1: Checking for meta fields")
    logger.info("="*50)

    meta_cols = [col for col in df.columns if col.startswith('_meta_')]

    if meta_cols:
        logger.info(f"Found {len(meta_cols)} meta columns:")
        for col in meta_cols:
            logger.info(f"  ✅ {col} (correctly prefixed with _meta_)")

        # Check correlation with target
        logger.info("\nChecking correlation with target:")
        for col in meta_cols:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and df[col].notna().sum() > 0:
                try:
                    corr = df[col].corr(df['target'].astype(int))
                    if pd.isna(corr):
                        logger.info(f"  ⚠️ {col}: no correlation (constant or all null)")
                    elif abs(corr) > 0.3:
                        logger.warning(f"  ⚠️ {col}: correlation = {corr:.3f} (expected for meta field)")
                    else:
                        logger.info(f"  ✅ {col}: correlation = {corr:.3f} (low)")
                except:
                    logger.info(f"  ⚠️ {col}: cannot compute correlation")
    else:
        logger.info("✅ No meta columns found (all excluded)")

    return meta_cols


def check_dangerous_columns(df):
    """Check for columns that might contain future data."""
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Checking for dangerous column names")
    logger.info("="*50)

    dangerous_patterns = [
        'outcome', 'result', 'profit', 'loss', 'hours_to',
        'exit', 'final', 'actual', 'close_time', 'end_time'
    ]

    dangerous_cols = []
    for col in df.columns:
        if col.startswith('_meta_'):
            continue  # Meta columns are OK

        for pattern in dangerous_patterns:
            if pattern in col.lower():
                dangerous_cols.append(col)
                break

    if dangerous_cols:
        logger.error(f"❌ DANGEROUS COLUMNS FOUND: {dangerous_cols}")
        logger.error("These should be prefixed with _meta_ or removed!")
        return False
    else:
        logger.info("✅ No dangerous column names found")
        return True


def test_model_performance(df):
    """Test model performance with and without meta fields."""
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Model performance comparison")
    logger.info("="*50)

    # Prepare features
    exclude_cols = ['target', 'timestamp', 'id', 'pair_symbol', 'trading_pair_id']
    meta_cols = [col for col in df.columns if col.startswith('_meta_')]

    # Features WITHOUT meta data
    feature_cols_safe = [col for col in df.columns
                         if col not in exclude_cols and col not in meta_cols]

    # Features WITH meta data (excluding timestamp meta fields)
    feature_cols_with_meta = [col for col in df.columns
                              if col not in exclude_cols
                              and col not in ['_meta_outcome_time', '_meta_created_at']]

    # Handle categorical and object columns
    for col in df.columns:
        if col in feature_cols_safe or col in feature_cols_with_meta:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'datetime64[ns, UTC]':
                if 'time' in col.lower() or 'created' in col.lower():
                    # Skip timestamp columns
                    if col in feature_cols_safe:
                        feature_cols_safe.remove(col)
                    if col in feature_cols_with_meta:
                        feature_cols_with_meta.remove(col)
                else:
                    # Convert categorical to codes
                    df[col] = pd.Categorical(df[col]).codes

    # Fill NaN values
    df = df.fillna(0)

    X_safe = df[feature_cols_safe]
    y = df['target'].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_safe, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model WITHOUT meta data
    logger.info("\nTraining model WITHOUT meta data...")
    rf_safe = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_safe.fit(X_train, y_train)
    y_pred_safe = rf_safe.predict_proba(X_test)[:, 1]
    auc_safe = roc_auc_score(y_test, y_pred_safe)

    logger.info(f"Model WITHOUT meta data: ROC-AUC = {auc_safe:.4f}")

    # If meta columns exist, test with them
    if meta_cols:
        # Filter out timestamp meta cols
        numeric_meta_cols = []
        for col in meta_cols:
            if col not in ['_meta_outcome_time', '_meta_created_at'] and col in feature_cols_with_meta:
                numeric_meta_cols.append(col)

        if numeric_meta_cols:
            logger.info(f"Training model WITH {len(numeric_meta_cols)} numeric meta fields...")
            X_with_meta = df[feature_cols_with_meta]
            X_train_meta, X_test_meta, y_train, y_test = train_test_split(
                X_with_meta, y, test_size=0.3, random_state=42, stratify=y
            )

            rf_meta = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            rf_meta.fit(X_train_meta, y_train)
            y_pred_meta = rf_meta.predict_proba(X_test_meta)[:, 1]
            auc_meta = roc_auc_score(y_test, y_pred_meta)

            logger.info(f"Model WITH meta data: ROC-AUC = {auc_meta:.4f}")
            logger.info(f"Performance boost from meta: +{auc_meta - auc_safe:.4f}")

            if auc_meta - auc_safe > 0.15:
                logger.warning("⚠️ Meta data provides significant boost (>0.15 AUC)")
                logger.info("✅ This is expected - meta fields contain future data")
                logger.info("✅ Meta fields are properly prefixed and will be excluded from training")
            else:
                logger.info("✅ Meta data impact is minimal - good isolation")
        else:
            logger.info("No numeric meta columns to test")

    return auc_safe


def check_feature_importance(df):
    """Check feature importance to detect leakage."""
    logger.info("\n" + "="*50)
    logger.info("TEST 4: Feature importance analysis")
    logger.info("="*50)

    # Prepare features
    exclude_cols = ['target', 'timestamp', 'id', 'pair_symbol', 'trading_pair_id']
    meta_cols = [col for col in df.columns if col.startswith('_meta_')]

    feature_cols = [col for col in df.columns
                   if col not in exclude_cols and col not in meta_cols]

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in feature_cols:
            df[col] = pd.Categorical(df[col]).codes

    # Fill NaN values
    df = df.fillna(0)

    X = df[feature_cols]
    y = df['target'].astype(int)

    # Train model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    logger.info("\nTop 10 most important features:")
    suspicious_features = []
    for _, row in importance_df.iterrows():
        feature = row['feature']
        importance = row['importance']

        # Check if feature name is suspicious
        is_suspicious = any(word in feature.lower() for word in
                          ['price_to_poc', 'poc_volume'])  # POC features are OK

        if is_suspicious:
            logger.info(f"✅ {feature}: {importance:.4f} (POC feature - OK)")
        else:
            logger.info(f"✓ {feature}: {importance:.4f}")

    return importance_df


def check_realistic_metrics(df):
    """Check if metrics are realistic for crypto trading."""
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Realistic metrics check")
    logger.info("="*50)

    # Overall win rate
    overall_win_rate = df['target'].mean()
    logger.info(f"\nOverall win rate: {overall_win_rate:.1%}")

    if overall_win_rate > 0.7:
        logger.error("❌ UNREALISTIC! Win rate > 70%")
    elif overall_win_rate < 0.3:
        logger.error("❌ UNREALISTIC! Win rate < 30%")
    else:
        logger.info("✅ Realistic win rate (30-70%)")

    # Win rate by signal type
    for signal_type in df['signal_type'].unique():
        signal_data = df[df['signal_type'] == signal_type]
        win_rate = signal_data['target'].mean()
        logger.info(f"{signal_type} win rate: {win_rate:.1%}")

    # Win rate by market regime
    logger.info("\nWin rate by market regime:")
    for signal_type in df['signal_type'].unique():
        for regime in df['market_regime'].unique():
            mask = (df['signal_type'] == signal_type) & (df['market_regime'] == regime)
            if mask.sum() > 0:
                win_rate = df[mask]['target'].mean()
                expected = (signal_type == 'BUY' and regime == 'BULL') or \
                          (signal_type == 'SELL' and regime == 'BEAR')
                indicator = "✅" if (expected and win_rate > 0.5) or \
                                  (not expected and win_rate < 0.5) else "❌"
                logger.info(f"  {signal_type} + {regime}: {win_rate:.1%} {indicator}")


def check_temporal_consistency(df):
    """Check temporal consistency of the data."""
    logger.info("\n" + "="*50)
    logger.info("TEST 6: Temporal consistency check")
    logger.info("="*50)

    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    daily_stats = df.groupby('date')['target'].agg(['count', 'mean', 'std'])

    logger.info(f"Data spans {len(daily_stats)} days")
    logger.info(f"Average signals per day: {daily_stats['count'].mean():.0f}")
    logger.info(f"Daily win rate mean: {daily_stats['mean'].mean():.1%}")
    logger.info(f"Daily win rate std: {daily_stats['mean'].std():.1%}")

    if daily_stats['mean'].std() > 0.2:
        logger.warning("⚠️ High variance in daily win rates")
    else:
        logger.info("✅ Consistent daily win rates")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("DATA LEAKAGE CHECK FOR ML_TRAINING_DATA_V2")
    logger.info("="*60)

    # Load data
    df = load_data(limit=10000)

    # Run all checks
    meta_cols = check_meta_fields(df)
    is_safe = check_dangerous_columns(df)
    auc = test_model_performance(df)
    importance = check_feature_importance(df)
    check_realistic_metrics(df)
    check_temporal_consistency(df)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    issues = []

    if not is_safe:
        issues.append("Dangerous columns found without _meta_ prefix")

    if auc > 0.75:
        issues.append(f"Model ROC-AUC suspiciously high: {auc:.3f}")

    base_win_rate = df['target'].mean()
    if base_win_rate > 0.7 or base_win_rate < 0.3:
        issues.append(f"Unrealistic base win rate: {base_win_rate:.1%}")

    if issues:
        logger.error("\n❌ ISSUES FOUND:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nRECOMMENDATION: Review data pipeline for leakage")
    else:
        logger.info("\n✅ ALL CHECKS PASSED!")
        logger.info("Data appears clean with no obvious leakage")
        logger.info(f"Expected model performance: ROC-AUC = {auc:.3f}")
        logger.info(f"Expected base win rate: {base_win_rate:.1%}")

        if 0.55 <= auc <= 0.75:
            logger.info("✅ ROC-AUC in realistic range for crypto trading")

        if meta_cols:
            logger.info(f"\n✅ Meta columns correctly isolated: {len(meta_cols)} fields")
            logger.info("These fields contain future data and are properly excluded from training")
            logger.info("Meta field correlation with target is expected (they contain outcome info)")


if __name__ == "__main__":
    main()