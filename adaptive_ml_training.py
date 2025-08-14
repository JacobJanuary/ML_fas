"""
Adaptive ML Training for Non-Stationary Data
==============================================
Использует sliding window и переобучается на последних N днях
для адаптации к быстро меняющемуся рынку.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import psycopg2
from datetime import datetime, timedelta
import joblib
import warnings
import logging
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveMLPipeline:
    """ML pipeline that adapts to changing market conditions."""

    def __init__(self, signal_type='BUY', window_days=7, min_train_samples=5000):
        """
        Args:
            signal_type: 'BUY' or 'SELL'
            window_days: Use only last N days for training
            min_train_samples: Minimum samples needed for training
        """
        self.signal_type = signal_type
        self.window_days = window_days
        self.min_train_samples = min_train_samples
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.threshold = 0.5

        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

    def load_recent_data(self, days_back=None, exclude_last_hours=24):
        """Load only recent data for training."""
        if days_back is None:
            days_back = self.window_days

        query = f"""
        SELECT *
        FROM fas.ml_training_data_direct
        WHERE signal_type = '{self.signal_type}'
            AND target IS NOT NULL
            AND timestamp >= NOW() - INTERVAL '{days_back} days'
            AND timestamp < NOW() - INTERVAL '{exclude_last_hours} hours'
            AND (NOW() - timestamp) > INTERVAL '48 hours' 
        ORDER BY timestamp
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} {self.signal_type} records from last {days_back} days")

        if len(df) > 0:
            # Analyze data stability
            daily_stats = df.groupby(pd.to_datetime(df['timestamp']).dt.date).agg({
                'target': ['count', 'mean']
            })

            win_rates = daily_stats['target']['mean'].values
            if len(win_rates) > 1:
                stability = 1 - np.std(win_rates)
                logger.info(f"Data stability: {stability:.2f} (1=stable, 0=unstable)")
                logger.info(f"Daily win rates: {[f'{wr:.1%}' for wr in win_rates]}")

        return df

    def prepare_features(self, df, is_training=True):
        """Prepare features with focus on recent patterns."""
        df = df.copy()

        # Remove dangerous columns
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                       'signal_type', 'signal_strength']
        remove_cols += [col for col in df.columns if col.startswith('_meta_')]

        for col in remove_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Fix extreme values
        for col in ['poc_volume_7d', 'poc_volume_24h']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=q99)
                df[f'{col}_log'] = np.log1p(df[col])
                # Drop original if too extreme
                if df[col].max() / (df[col].quantile(0.95) + 1) > 100:
                    df = df.drop(columns=[col])

        # Add time-based features for adaptation
        df['timestamp_hour'] = pd.to_datetime(df.index).hour
        df['timestamp_day'] = pd.to_datetime(df.index).dayofweek

        # Add rolling features (if enough data)
        if len(df) > 100:
            # Recent performance indicators
            df['recent_avg'] = df['target'].rolling(window=50, min_periods=10).mean()
            df['recent_std'] = df['target'].rolling(window=50, min_periods=10).std()
            df.fillna(method='bfill', inplace=True)

        # Market regime features
        if 'market_regime' in df.columns:
            # One-hot encoding
            df['regime_bull'] = (df['market_regime'] == 'BULL').astype(float)
            df['regime_bear'] = (df['market_regime'] == 'BEAR').astype(float)
            df['regime_neutral'] = (df['market_regime'] == 'NEUTRAL').astype(float)

            # Alignment features
            if self.signal_type == 'BUY':
                df['good_alignment'] = df['regime_bull']
                df['bad_alignment'] = df['regime_bear']
            else:
                df['good_alignment'] = df['regime_bear']
                df['bad_alignment'] = df['regime_bull']

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'target':
                df[col] = pd.Categorical(df[col].fillna('unknown')).codes

        # Remove constant features
        if is_training:
            constant_features = []
            for col in df.columns:
                if col != 'target' and df[col].nunique() <= 1:
                    constant_features.append(col)

            if constant_features:
                df = df.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features")

            self.feature_columns = [col for col in df.columns if col != 'target']
            logger.info(f"Using {len(self.feature_columns)} features")

        # Prepare X and y
        X = df[self.feature_columns].fillna(0)
        y = df['target'].astype(int) if 'target' in df.columns else None

        # Scale features
        if is_training:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return X_scaled, y

    def train_adaptive_model(self):
        """Train model on recent data with cross-validation."""
        # Load recent data
        df = self.load_recent_data()

        if len(df) < self.min_train_samples:
            logger.warning(f"Not enough data: {len(df)} < {self.min_train_samples}")
            return None

        # Prepare features
        X, y = self.prepare_features(df, is_training=True)

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)

        best_model = None
        best_score = -1
        best_params = None

        # Try different parameter sets
        param_sets = [
            {
                'name': 'Conservative',
                'params': {
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 20
                }
            },
            {
                'name': 'Moderate',
                'params': {
                    'learning_rate': 0.1,
                    'max_depth': 4,
                    'n_estimators': 150,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'min_child_weight': 10
                }
            },
            {
                'name': 'Aggressive',
                'params': {
                    'learning_rate': 0.15,
                    'max_depth': 5,
                    'n_estimators': 100,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'min_child_weight': 5
                }
            }
        ]

        logger.info("\nTesting different model configurations...")

        for param_set in param_sets:
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                model = xgb.XGBClassifier(
                    **param_set['params'],
                    objective='binary:logistic',
                    eval_metric='logloss',
                    random_state=42,
                    verbosity=0
                )

                model.fit(X_train, y_train)

                # Evaluate
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # Find best threshold for 20-30% signals
                thresholds = np.percentile(y_pred_proba, [70, 75, 80])
                best_val_score = 0

                for thresh in thresholds:
                    y_pred = (y_pred_proba >= thresh).astype(int)
                    if y_pred.sum() > 0:
                        trades_pct = y_pred.sum() / len(y_pred)
                        if 0.15 <= trades_pct <= 0.35:
                            win_rate = y_val[y_pred == 1].mean()
                            score = win_rate * trades_pct
                            if score > best_val_score:
                                best_val_score = score

                scores.append(best_val_score)

            avg_score = np.mean(scores)
            logger.info(f"  {param_set['name']}: avg score = {avg_score:.3f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = param_set['params']
                best_model = param_set['name']

        # Train final model on all data
        logger.info(f"\nTraining final model ({best_model})...")
        self.model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

        self.model.fit(X, y)

        # Find optimal threshold
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        self.threshold = self.find_optimal_threshold(y_pred_proba, y)

        # Evaluate on training data
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        trades_pct = y_pred.sum() / len(y_pred)
        win_rate = y[y_pred == 1].mean() if y_pred.sum() > 0 else 0

        logger.info(f"\nTraining performance:")
        logger.info(f"  Threshold: {self.threshold:.3f}")
        logger.info(f"  Trades: {trades_pct:.1%} of signals")
        logger.info(f"  Win rate: {win_rate:.1%}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("\nTop 10 features:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        return self.model

    def find_optimal_threshold(self, y_pred_proba, y_true):
        """Find threshold for best performance on 20-30% of signals."""
        best_threshold = 0.5
        best_score = 0

        for pct in [70, 72, 75, 77, 80]:
            threshold = np.percentile(y_pred_proba, pct)
            y_pred = (y_pred_proba >= threshold).astype(int)

            if y_pred.sum() > 0:
                trades_pct = y_pred.sum() / len(y_pred)
                if 0.15 <= trades_pct <= 0.35:
                    win_rate = y_true[y_pred == 1].mean()
                    score = win_rate * trades_pct

                    if score > best_score:
                        best_score = score
                        best_threshold = threshold

        return best_threshold

    def predict_next_signals(self, hours_ahead=24):
        """Predict on upcoming signals."""
        if self.model is None:
            logger.error("No model trained yet!")
            return None

        query = f"""
        SELECT *
        FROM fas.ml_training_data_direct
        WHERE signal_type = '{self.signal_type}'
            AND timestamp >= NOW() - INTERVAL '{hours_ahead} hours'
        ORDER BY timestamp DESC
        LIMIT 1000
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn)

        if len(df) == 0:
            logger.warning("No recent signals to predict")
            return None

        # Prepare features
        X, _ = self.prepare_features(df, is_training=False)

        # Predict
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Add predictions to dataframe
        df['prediction_proba'] = y_pred_proba
        df['prediction'] = y_pred

        # Return only predicted positive signals
        positive_signals = df[df['prediction'] == 1].copy()

        logger.info(f"\nPredictions for next {hours_ahead} hours:")
        logger.info(f"  Total signals: {len(df)}")
        logger.info(f"  Positive predictions: {len(positive_signals)} ({len(positive_signals) / len(df):.1%})")

        if len(positive_signals) > 0:
            logger.info(f"  Avg confidence: {positive_signals['prediction_proba'].mean():.3f}")

            # Show top signals
            top_signals = positive_signals.nlargest(5, 'prediction_proba')
            logger.info("\nTop 5 signals:")
            for _, signal in top_signals.iterrows():
                logger.info(f"  {signal.get('pair_symbol', 'Unknown')}: "
                            f"prob={signal['prediction_proba']:.3f}, "
                            f"score={signal.get('total_score', 0):.1f}")

        return positive_signals

    def save_model(self, path='models/'):
        """Save adaptive model."""
        os.makedirs(path, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns,
            'signal_type': self.signal_type,
            'window_days': self.window_days,
            'timestamp': datetime.now().isoformat()
        }

        filename = f'{path}{self.signal_type.lower()}_adaptive_model.pkl'
        joblib.dump(model_data, filename)
        logger.info(f"Model saved to {filename}")


def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("ADAPTIVE ML TRAINING FOR NON-STATIONARY DATA")
    logger.info("=" * 60)

    # Test different window sizes
    window_sizes = [5, 7, 10]

    best_results = {}

    for window in window_sizes:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Testing {window}-day window")
        logger.info(f"{'=' * 40}")

        # Train BUY model
        logger.info("\n📈 Training BUY model...")
        buy_pipeline = AdaptiveMLPipeline('BUY', window_days=window)
        buy_model = buy_pipeline.train_adaptive_model()

        if buy_model:
            buy_predictions = buy_pipeline.predict_next_signals()
            best_results[f'BUY_{window}d'] = buy_pipeline

        # Train SELL model
        logger.info("\n📉 Training SELL model...")
        sell_pipeline = AdaptiveMLPipeline('SELL', window_days=window)
        sell_model = sell_pipeline.train_adaptive_model()

        if sell_model:
            sell_predictions = sell_pipeline.predict_next_signals()
            best_results[f'SELL_{window}d'] = sell_pipeline

    # Save best models
    logger.info("\n" + "=" * 60)
    logger.info("SAVING BEST MODELS")
    logger.info("=" * 60)

    for name, pipeline in best_results.items():
        pipeline.save_model()
        logger.info(f"Saved: {name}")

    logger.info("\n✅ Adaptive models trained and saved")
    logger.info("⚠️ Re-train daily or weekly due to data instability")


if __name__ == "__main__":
    main()