"""
Feature preprocessing utilities for multi-stage models.
Based on original adaptive_ml_training.py logic.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Preprocess features for model training, handling POC and categorical data."""

    @staticmethod
    def prepare_features(df, is_training=True, model_type='full'):
        """
        Prepare features with proper handling of POC and categorical columns.

        Args:
            df: Input dataframe
            is_training: Whether this is for training (affects feature selection)
            model_type: 'quick_filter', 'regime_specific', or 'precision'
        """
        df = df.copy()

        # Remove ID and metadata columns
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                       'signal_type', '_meta_outcome_time', '_meta_created_at',
                       '_meta_outcome_type']

        for col in remove_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # CRITICAL: Handle POC columns (they appear as object but are actually numeric with NaN)
        poc_columns = ['poc_volume_24h', 'poc_volume_7d', 'price_to_poc_24h_pct',
                       'price_to_poc_7d_pct', 'price_to_poc_30d_pct', 'rs_momentum']

        for col in poc_columns:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # For POC volume columns, handle extreme values
                if 'volume' in col:
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0)

                    # Clip extreme values if we have enough data
                    if len(df) > 100:
                        q99 = df[col].quantile(0.99)
                        if q99 > 0:
                            df[col] = df[col].clip(upper=q99)

                    # Add log transform
                    df[f'{col}_log'] = np.log1p(df[col])

                    # Check if too extreme and drop if necessary
                    if len(df) > 100:
                        q95 = df[col].quantile(0.95)
                        if q95 > 0 and df[col].max() / (q95 + 1) > 100:
                            logger.info(f"  Dropping {col} due to extreme values")
                            df = df.drop(columns=[col])
                else:
                    # For percentage columns, just fill NaN
                    df[col] = df[col].fillna(0)

        # Handle categorical columns by encoding them
        categorical_mappings = {
            'signal_strength': {'WEAK': 1, 'MODERATE': 2, 'STRONG': 3, 'VERY_STRONG': 4},
            'market_regime': {'BEAR': -1, 'NEUTRAL': 0, 'BULL': 1},
            'pattern_1_name': {
                'ACCUMULATION': 1, 'DISTRIBUTION': -1, 'VOLUME_ANOMALY': 2,
                'MOMENTUM_EXHAUSTION': 3, 'OI_EXPLOSION': 4, 'SQUEEZE_IGNITION': 5,
                'CVD_PRICE_DIVERGENCE': 6, 'SMART_MONEY_DIVERGENCE': 7,
                'FUNDING_DIVERGENCE': 8
            }
        }

        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                # Map values, use 0 for unknown/NaN
                df[col + '_encoded'] = df[col].map(mapping).fillna(0)
                df = df.drop(columns=[col])

        # Handle pattern and combo names - create binary indicators
        pattern_cols = ['pattern_1_name', 'pattern_2_name', 'pattern_3_name']
        for col in pattern_cols:
            if col in df.columns:
                # Already handled above if it's pattern_1_name
                if col != 'pattern_1_name':
                    # Create binary: has pattern or not
                    df[col + '_exists'] = df[col].notna().astype(int)
                    df = df.drop(columns=[col])

        combo_cols = ['combo_1_name', 'combo_2_name']
        for col in combo_cols:
            if col in df.columns:
                df[col + '_exists'] = df[col].notna().astype(int)
                df = df.drop(columns=[col])

        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if col != 'target':  # Keep target as is if it exists
                df[col] = df[col].astype(int)

        # Handle remaining object columns
        remaining_object_cols = df.select_dtypes(include=['object']).columns
        for col in remaining_object_cols:
            if col != 'target':
                # Try to convert to numeric
                numeric_attempt = pd.to_numeric(df[col], errors='coerce')
                if numeric_attempt.notna().sum() > len(df) * 0.5:  # More than 50% are numeric
                    df[col] = numeric_attempt.fillna(0)
                else:
                    # Drop if can't convert
                    logger.info(f"  Dropping non-convertible column: {col}")
                    df = df.drop(columns=[col])

        # Add time-based features if needed
        if 'timestamp' in df.index.names or df.index.dtype == 'datetime64[ns]':
            df['hour'] = pd.to_datetime(df.index).hour
            df['dayofweek'] = pd.to_datetime(df.index).dayofweek

        # Market regime interaction features (important for adaptation)
        if 'market_regime_encoded' in df.columns:
            if 'total_score' in df.columns:
                df['regime_score_interaction'] = df['market_regime_encoded'] * df['total_score']
            if 'rsi' in df.columns:
                df['regime_rsi_interaction'] = df['market_regime_encoded'] * df['rsi']

        # Feature selection based on model type
        if model_type == 'quick_filter':
            # Use only essential features for speed
            essential_features = [
                'total_score', 'rsi', 'volume_zscore', 'price_change_pct',
                'buy_ratio', 'cvd_delta', 'signal_strength_encoded',
                'market_regime_encoded', 'pattern_count', 'combo_count'
            ]
            available_features = [f for f in essential_features if f in df.columns]

            # Keep only essential features and target
            keep_cols = available_features
            if 'target' in df.columns:
                keep_cols.append('target')

            df = df[keep_cols]

        elif model_type == 'regime_specific':
            # Use most features but not meta fields
            exclude_patterns = ['_meta_', 'combo_2_', 'pattern_3_']
            for pattern in exclude_patterns:
                cols_to_drop = [col for col in df.columns if pattern in col]
                df = df.drop(columns=cols_to_drop, errors='ignore')

        # Remove constant features if training
        if is_training:
            constant_features = []
            for col in df.columns:
                if col != 'target' and df[col].nunique() <= 1:
                    constant_features.append(col)

            if constant_features:
                df = df.drop(columns=constant_features)
                logger.info(f"  Removed {len(constant_features)} constant features")

        # Final cleanup - ensure all remaining columns are numeric
        for col in df.columns:
            if col != 'target' and df[col].dtype == 'object':
                logger.warning(f"  Unexpected object column remains: {col}")
                df = df.drop(columns=[col])

        # Fill any remaining NaN
        df = df.fillna(0)

        # Ensure no infinity values
        df = df.replace([np.inf, -np.inf], 0)

        return df


def test_preprocessing():
    """Test the preprocessing on sample data."""
    import psycopg2
    from dotenv import load_dotenv
    import os

    load_dotenv()

    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    # Load sample data
    query = """
    SELECT *
    FROM fas.ml_training_data_direct
    WHERE signal_type = 'BUY'
    LIMIT 1000
    """

    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)

    print("Original shape:", df.shape)
    print("Original columns:", len(df.columns))

    # Test preprocessing
    preprocessor = FeaturePreprocessor()

    # Test for different model types
    for model_type in ['quick_filter', 'regime_specific', 'precision']:
        print(f"\n{model_type.upper()} Model:")
        df_processed = preprocessor.prepare_features(df.copy(), is_training=True, model_type=model_type)

        # Separate features and target
        if 'target' in df_processed.columns:
            X = df_processed.drop('target', axis=1)
            y = df_processed['target']
        else:
            X = df_processed
            y = None

        print(f"  Features shape: {X.shape}")
        print(f"  Feature columns: {X.shape[1]}")
        print(f"  All numeric: {all(X.dtypes != 'object')}")
        print(f"  Has NaN: {X.isna().any().any()}")
        print(f"  Has Inf: {np.isinf(X.select_dtypes(include=[np.number])).any().any()}")

        # Show top features
        if X.shape[1] > 0:
            print(f"  Sample features: {list(X.columns[:10])}")


if __name__ == "__main__":
    test_preprocessing()