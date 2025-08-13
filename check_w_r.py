"""
ML Model Training Script for Trading Signals - UPDATED FOR ml_training_data_v2
===============================================================================
Updated to work with new view structure with proper POC levels and realistic targets
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from optuna.samplers import TPESampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class DatabaseConnection:
    """Handle PostgreSQL database connections."""

    def __init__(self):
        """Initialize database connection parameters from environment."""
        load_dotenv()
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

    def get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(**self.connection_params)

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            return pd.read_sql(query, conn)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering."""

    def __init__(self, signal_type: str):
        """
        Initialize preprocessor for specific signal type.

        Args:
            signal_type: 'BUY' or 'SELL'
        """
        self.signal_type = signal_type
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def load_data(self, db: DatabaseConnection) -> pd.DataFrame:
        """
        Load training data from the new ml_training_data_v2 view.

        Args:
            db: Database connection instance

        Returns:
            DataFrame with training data
        """
        query = f"""
        SELECT *
        FROM fas.ml_training_data_v2
        WHERE signal_type = '{self.signal_type}'
            AND target IS NOT NULL
            AND timestamp < NOW() - INTERVAL '12 hours'
        ORDER BY timestamp
        """

        logger.info(f"Loading {self.signal_type} data from ml_training_data_v2...")
        df = db.fetch_data(query)
        logger.info(f"Loaded {len(df)} records for {self.signal_type}")

        # Data quality checks
        win_rate = df['target'].mean()
        logger.info(f"Base win rate: {win_rate:.1%}")

        if win_rate > 0.7:
            logger.warning(f"⚠️ Win rate {win_rate:.1%} seems high!")
        elif win_rate < 0.3:
            logger.warning(f"⚠️ Win rate {win_rate:.1%} seems low!")
        else:
            logger.info(f"✅ Win rate {win_rate:.1%} looks realistic")

        # Market regime distribution
        logger.info(f"Market regime distribution:")
        logger.info(f"{df['market_regime'].value_counts()}")

        # Pattern distribution
        if 'pattern_count' in df.columns:
            logger.info(f"Pattern count distribution:")
            logger.info(f"{df['pattern_count'].value_counts().head()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features optimized for new structure.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # ========== MARKET REGIME FEATURES (HIGH IMPORTANCE) ==========
        df['regime_signal_match'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BULL')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BEAR'))
        ).astype(float) * 10  # Weight for importance

        df['regime_conflict'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BEAR')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BULL'))
        ).astype(float) * -10  # Negative weight

        df['regime_neutral'] = (df['market_regime'] == 'NEUTRAL').astype(float) * 5

        # Combine regime with signal strength (already numeric in new view)
        if 'strength_numeric' in df.columns:
            df['strong_signal_right_regime'] = df['regime_signal_match'] * (df['strength_numeric'] >= 3)
            df['weak_signal_wrong_regime'] = df['regime_conflict'] * (df['strength_numeric'] <= 2)

        # Numeric regime encoding
        regime_weights = {'BULL': 1.0, 'NEUTRAL': 0.0, 'BEAR': -1.0}
        df['regime_numeric'] = df['market_regime'].map(regime_weights)

        # Regime interactions with key indicators
        df['regime_x_total_score'] = df['regime_numeric'] * df['total_score']
        df['regime_x_rsi'] = df['regime_numeric'] * df['rsi']
        df['regime_x_volume'] = df['regime_numeric'] * df['volume_zscore']

        # ========== POC FEATURES (already relative in new view) ==========
        # Create POC divergence features
        if 'price_to_poc_24h_pct' in df.columns:
            df['poc_divergence'] = df[['price_to_poc_24h_pct', 'price_to_poc_7d_pct', 'price_to_poc_30d_pct']].std(axis=1)
            df['poc_mean_distance'] = df[['price_to_poc_24h_pct', 'price_to_poc_7d_pct', 'price_to_poc_30d_pct']].mean(axis=1)

            # POC trend (24h vs 30d)
            df['poc_trend'] = df['price_to_poc_24h_pct'] - df['price_to_poc_30d_pct']

        # ========== PATTERN & COMBO FEATURES ==========
        # Pattern intensity (using new pattern_count field)
        if 'pattern_count' in df.columns:
            df['multi_pattern_signal'] = (df['pattern_count'] > 1).astype(float)
            df['pattern_concentration'] = df['pattern_score'] / (df['pattern_count'] + 1e-8)

        # Pattern confidence aggregation
        pattern_conf_cols = [col for col in df.columns if 'pattern_' in col and 'confidence' in col]
        if pattern_conf_cols:
            df['max_pattern_confidence'] = df[pattern_conf_cols].max(axis=1)
            df['avg_pattern_confidence'] = df[pattern_conf_cols].mean(axis=1)

        # Combo features (using new combo_count field)
        if 'combo_count' in df.columns:
            df['has_combo'] = (df['combo_count'] > 0).astype(float)
            df['combo_intensity'] = df['combination_score'] / (df['combo_count'] + 1e-8)

        # ========== SCORE FEATURES ==========
        df['pattern_to_indicator_ratio'] = df['pattern_score'] / (df['indicator_score'] + 1e-8)
        df['combo_to_total_ratio'] = df['combination_score'] / (df['total_score'] + 1e-8)

        # Score distribution
        df['score_concentration'] = df[['indicator_score', 'pattern_score', 'combination_score']].std(axis=1)

        # ========== VOLUME FEATURES ==========
        df['volume_pressure'] = df['buy_ratio_weighted'] * df['volume_zscore']
        df['cvd_momentum'] = df['cvd_delta'] / (abs(df['cvd_cumulative']) + 1e-8)

        # Volume regime alignment
        df['volume_regime_alignment'] = (
            ((df['signal_type'] == 'BUY') & (df['buy_ratio'] > 0.5)) |
            ((df['signal_type'] == 'SELL') & (df['buy_ratio'] < 0.5))
        ).astype(float)

        # ========== VOLATILITY FEATURES ==========
        # ATR% already in new view, create additional features
        if 'atr_pct' in df.columns:
            df['score_per_volatility'] = df['total_score'] / (df['atr_pct'] + 1e-8)
            df['high_volatility'] = (df['atr_pct'] > df['atr_pct'].quantile(0.75)).astype(float)

        # ========== TEMPORAL FEATURES ==========
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

        # Weekend flag
        df['is_weekend'] = (df['dow'] >= 5).astype(float)

        logger.info(f"Engineered {len(df.columns) - len(df.columns)} new features")

        return df

    def create_sample_weights(self, df: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Create sample weights based on market regime alignment.

        Args:
            df: Feature DataFrame
            y: Target Series

        Returns:
            Array of sample weights
        """
        weights = np.ones(len(df))

        # Weight based on regime alignment
        bull_buy_mask = (df['signal_type'] == 'BUY') & (df['market_regime'] == 'BULL')
        weights[bull_buy_mask] = 1.5

        bear_sell_mask = (df['signal_type'] == 'SELL') & (df['market_regime'] == 'BEAR')
        weights[bear_sell_mask] = 1.5

        bear_buy_mask = (df['signal_type'] == 'BUY') & (df['market_regime'] == 'BEAR')
        weights[bear_buy_mask] = 0.5

        bull_sell_mask = (df['signal_type'] == 'SELL') & (df['market_regime'] == 'BULL')
        weights[bull_sell_mask] = 0.5

        # Extra weight for successful trades in right regime
        successful_trades = y == 1
        weights[bull_buy_mask & successful_trades] = 2.0
        weights[bear_sell_mask & successful_trades] = 2.0

        logger.info(f"Sample weights distribution:")
        logger.info(f"  Weight 2.0 (best): {(weights == 2.0).sum()} samples")
        logger.info(f"  Weight 1.5 (good): {(weights == 1.5).sum()} samples")
        logger.info(f"  Weight 1.0 (neutral): {(weights == 1.0).sum()} samples")
        logger.info(f"  Weight 0.5 (bad): {(weights == 0.5).sum()} samples")

        return weights

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training with new view structure.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fit encoders)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.engineer_features(df)

        # Separate target (now called 'target' in new view)
        y = df['target'].astype(int)

        # Handle categorical features
        categorical_cols = ['market_regime', 'pattern_1_name', 'pattern_2_name',
                          'pattern_3_name', 'combo_1_name', 'combo_2_name']

        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = df[col].fillna('unknown')
                    df[f'{col}_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    df[col] = df[col].fillna('unknown')
                    le = self.label_encoders.get(col)
                    if le:
                        df[f'{col}_encoded'] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

        # Select feature columns
        if is_training:
            # Exclude non-feature columns
            exclude_cols = [
                'id', 'timestamp', 'trading_pair_id', 'pair_symbol',
                'target', 'signal_type', 'signal_strength', 'is_meme'
            ] + categorical_cols  # Original categorical columns (we use encoded versions)

            # CRITICAL: Exclude all meta columns
            meta_cols = [col for col in df.columns if col.startswith('_meta_')]
            if meta_cols:
                logger.info(f"✅ Excluding {len(meta_cols)} meta columns from training")
                exclude_cols.extend(meta_cols)

            # Safety check for any remaining dangerous columns
            dangerous_patterns = ['outcome', 'result', 'profit', 'loss', 'hours_to']
            dangerous_cols = [col for col in df.columns
                            if any(pattern in col.lower() for pattern in dangerous_patterns)
                            and col not in exclude_cols]

            if dangerous_cols:
                logger.warning(f"⚠️ Found and excluding dangerous columns: {dangerous_cols}")
                exclude_cols.extend(dangerous_cols)

            self.feature_columns = [col for col in df.columns if col not in exclude_cols]

            logger.info(f"Selected {len(self.feature_columns)} features for training")

            # Log sample of selected features
            logger.info(f"Sample features: {self.feature_columns[:10]}")

        # Select features
        X = df[self.feature_columns]

        # Handle missing values intelligently
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().any():
                null_count = X[col].isnull().sum()
                if null_count < len(X) * 0.5:  # Less than 50% missing
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                else:
                    # Too many missing, fill with 0
                    X[col] = X[col].fillna(0)

        # Scale numerical features
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


class ModelOptimizer:
    """Optimize hyperparameters using Optuna."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 sample_weights: np.ndarray = None):
        """
        Initialize optimizer with training and validation data.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            sample_weights: Optional sample weights for training
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weights = sample_weights if sample_weights is not None else np.ones(len(X_train))

    def optimize_xgboost(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Best parameters dictionary
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 30,
                'enable_categorical': False
            }

            try:
                model = xgb.XGBClassifier(**params)
                model.fit(
                    self.X_train, self.y_train,
                    sample_weight=self.sample_weights,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False
                )

                y_pred = model.predict_proba(self.X_val)[:, 1]
                auc = roc_auc_score(self.y_val, y_pred)

                # Penalize overly optimistic scores
                if auc > 0.75:
                    auc = auc * 0.95

                return auc
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.5

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"XGBoost best AUC: {study.best_value:.4f}")

        best_params = study.best_params.copy()
        best_params['early_stopping_rounds'] = 30
        best_params['enable_categorical'] = False
        return best_params

    def optimize_lightgbm(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Best parameters dictionary
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 10, 60),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.8),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                self.X_train, self.y_train,
                sample_weight=self.sample_weights,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            y_pred = model.predict_proba(self.X_val)[:, 1]
            auc = roc_auc_score(self.y_val, y_pred)

            if auc > 0.75:
                auc = auc * 0.95

            return auc

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"LightGBM best AUC: {study.best_value:.4f}")
        return study.best_params

    def optimize_catboost(self, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize CatBoost hyperparameters.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Best parameters dictionary
        """
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 128),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 0.5),
                'random_strength': trial.suggest_float('random_strength', 0, 0.5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2),
                'random_state': 42,
                'verbose': False,
                'thread_count': -1
            }

            model = CatBoostClassifier(**params)
            model.fit(
                self.X_train, self.y_train,
                sample_weight=self.sample_weights,
                eval_set=(self.X_val, self.y_val),
                early_stopping_rounds=30,
                verbose=False
            )

            y_pred = model.predict_proba(self.X_val)[:, 1]
            auc = roc_auc_score(self.y_val, y_pred)

            if auc > 0.75:
                auc = auc * 0.95

            return auc

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"CatBoost best AUC: {study.best_value:.4f}")
        return study.best_params


class ModelEvaluator:
    """Evaluate and compare model performance."""

    def __init__(self, signal_type: str):
        """
        Initialize evaluator.

        Args:
            signal_type: 'BUY' or 'SELL'
        """
        self.signal_type = signal_type
        self.results = {}

    def evaluate_model(self, model: Any, X_test: pd.DataFrame,
                       y_test: pd.Series, model_name: str,
                       threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate single model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            threshold: Probability threshold for classification

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': (y_pred == y_test).mean(),
            'threshold': threshold
        }

        # Calculate expected profit
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        total_predictions = (y_pred == 1).sum()

        if total_predictions > 0:
            metrics['expected_profit'] = (tp * 3 - fp * 3) / total_predictions
            metrics['win_rate'] = tp / total_predictions
            metrics['trades_taken'] = total_predictions / len(y_test)
        else:
            metrics['expected_profit'] = 0
            metrics['win_rate'] = 0
            metrics['trades_taken'] = 0

        self.results[model_name] = metrics

        logger.info(f"\n{model_name} Performance (threshold={threshold:.2f}):")
        for metric, value in metrics.items():
            if metric == 'trades_taken':
                logger.info(f"  {metric}: {value:.1%} of signals")
            elif metric in ['win_rate', 'precision', 'recall', 'accuracy']:
                logger.info(f"  {metric}: {value:.1%}")
            else:
                logger.info(f"  {metric}: {value:.4f}")

        # Quality check
        if 0.60 <= metrics['win_rate'] <= 0.70 and 0.15 <= metrics['trades_taken'] <= 0.35:
            logger.info(f"✅ GOOD: Realistic performance - {metrics['win_rate']:.1%} win rate on {metrics['trades_taken']:.1%} of signals")
        elif metrics['roc_auc'] > 0.70:
            logger.warning(f"⚠️ ROC-AUC {metrics['roc_auc']:.3f} seems high for crypto trading")

        return metrics

    def find_optimal_threshold(self, model: Any, X_val: pd.DataFrame,
                              y_val: pd.Series, target_precision: float = 0.65) -> float:
        """
        Find optimal probability threshold to achieve target precision.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            target_precision: Target precision to achieve

        Returns:
            Optimal threshold
        """
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        best_threshold = 0.5
        best_score = 0

        for threshold in np.arange(0.3, 0.8, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)

            if y_pred.sum() == 0:
                continue

            precision = precision_score(y_val, y_pred, zero_division=0)
            trades_taken = y_pred.sum() / len(y_pred)

            if precision >= target_precision and trades_taken >= 0.1:
                score = precision * trades_taken
                if score > best_score:
                    best_score = score
                    best_threshold = threshold

        logger.info(f"Optimal threshold found: {best_threshold:.2f}")
        return best_threshold

    def plot_comparison(self, save_path: str = None):
        """
        Plot model comparison.

        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.signal_type} Models Comparison', fontsize=16)

        metrics = ['roc_auc', 'precision', 'win_rate', 'trades_taken', 'expected_profit', 'f1']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            models = list(self.results.keys())
            values = [self.results[m][metric] for m in models]

            bars = ax.bar(models, values)
            ax.set_title(metric.replace('_', ' ').title())

            if metric in ['trades_taken', 'win_rate', 'precision']:
                ax.set_ylim([0, 1])
                if metric == 'win_rate':
                    ax.axhspan(0.60, 0.70, alpha=0.2, color='green', label='Target')
                elif metric == 'trades_taken':
                    ax.axhspan(0.15, 0.35, alpha=0.2, color='green', label='Target')
            elif metric == 'expected_profit':
                ax.set_ylim([min(values) - 0.5, max(values) + 0.5])

            best_idx = np.argmax(values) if metric != 'trades_taken' else np.argmin(np.abs(np.array(values) - 0.25))
            bars[best_idx].set_color('green')

            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric in ['trades_taken', 'win_rate', 'precision']:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1%}', ha='center', va='bottom')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()

    def get_best_model(self) -> str:
        """
        Get the name of the best model based on balanced metrics.

        Returns:
            Name of the best model
        """
        if not self.results:
            return None

        best_score = -1
        best_model = None

        for model_name, metrics in self.results.items():
            if metrics['win_rate'] >= 0.55 and metrics['trades_taken'] >= 0.15:
                score = (
                    metrics['win_rate'] * 0.4 +
                    min(metrics['trades_taken'], 0.3) * 0.3 +
                    max(metrics['expected_profit'] / 3, 0) * 0.3
                )

                if score > best_score:
                    best_score = score
                    best_model = model_name

        if best_model:
            logger.info(f"\nBest model for {self.signal_type}: {best_model}")
            logger.info(f"Win rate: {self.results[best_model]['win_rate']:.1%}")
            logger.info(f"Trades taken: {self.results[best_model]['trades_taken']:.1%}")
            logger.info(f"Expected profit: {self.results[best_model]['expected_profit']:.2%}")
        else:
            logger.warning("No model meets minimum criteria")
            best_model = list(self.results.keys())[0] if self.results else None

        return best_model


class TradingModelPipeline:
    """Main pipeline for training trading models."""

    def __init__(self, signal_type: str = 'BUY'):
        """
        Initialize pipeline.

        Args:
            signal_type: 'BUY' or 'SELL'
        """
        self.signal_type = signal_type
        self.db = DatabaseConnection()
        self.preprocessor = DataPreprocessor(signal_type)
        self.models = {}
        self.best_model = None
        self.optimal_threshold = 0.5

    def train_and_evaluate(self, optimize_hyperparams: bool = True,
                          n_trials: int = 30) -> Dict[str, Any]:
        """
        Train and evaluate all models.

        Args:
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials

        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {self.signal_type} Models")
        logger.info(f"{'='*50}\n")

        # Load and prepare data
        df = self.preprocessor.load_data(self.db)

        # Sort by timestamp for temporal split
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save original data for analysis
        df_original = df.copy()

        X, y = self.preprocessor.prepare_features(df, is_training=True)

        # Create sample weights
        sample_weights = self.preprocessor.create_sample_weights(df_original, y)

        # Temporal split
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        weights_train = sample_weights[:train_size]
        df_train = df_original.iloc[:train_size]

        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        weights_val = sample_weights[train_size:train_size + val_size]
        df_val = df_original.iloc[train_size:train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        weights_test = sample_weights[train_size + val_size:]
        df_test = df_original.iloc[train_size + val_size:]

        logger.info(f"Data split (TEMPORAL):")
        logger.info(f"  Train: {len(X_train)} samples (first 70%)")
        logger.info(f"  Val: {len(X_val)} samples (next 15%)")
        logger.info(f"  Test: {len(X_test)} samples (last 15%)")
        logger.info(f"  Train period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
        logger.info(f"  Val period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
        logger.info(f"  Test period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

        logger.info(f"\nTarget distribution:")
        logger.info(f"  Train: {y_train.mean():.2%} win rate")
        logger.info(f"  Val: {y_val.mean():.2%} win rate")
        logger.info(f"  Test: {y_test.mean():.2%} win rate")

        # Initialize evaluator
        evaluator = ModelEvaluator(self.signal_type)

        if optimize_hyperparams:
            optimizer = ModelOptimizer(X_train, y_train, X_val, y_val, weights_train)

            logger.info("\nOptimizing XGBoost...")
            xgb_params = optimizer.optimize_xgboost(n_trials)

            logger.info("\nOptimizing LightGBM...")
            lgb_params = optimizer.optimize_lightgbm(n_trials)

            logger.info("\nOptimizing CatBoost...")
            cb_params = optimizer.optimize_catboost(min(n_trials, 20))
        else:
            # Conservative default parameters
            xgb_params = {
                'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'random_state': 42,
                'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1,
                'early_stopping_rounds': 30, 'enable_categorical': False
            }
            lgb_params = {
                'n_estimators': 100, 'num_leaves': 20, 'learning_rate': 0.05,
                'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'random_state': 42,
                'lambda_l1': 1, 'lambda_l2': 1, 'verbose': -1
            }
            cb_params = {
                'iterations': 100, 'depth': 4, 'learning_rate': 0.05,
                'l2_leaf_reg': 3, 'random_state': 42, 'verbose': False
            }

        # Train final models
        logger.info("\nTraining final models...")

        # XGBoost
        self.models['XGBoost'] = xgb.XGBClassifier(**xgb_params)
        self.models['XGBoost'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            verbose=0
        )

        xgb_threshold = evaluator.find_optimal_threshold(
            self.models['XGBoost'], X_val, y_val, target_precision=0.65
        )
        evaluator.evaluate_model(self.models['XGBoost'], X_test, y_test, 'XGBoost', threshold=xgb_threshold)

        # LightGBM
        self.models['LightGBM'] = lgb.LGBMClassifier(**lgb_params, verbose=-1)
        self.models['LightGBM'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        lgb_threshold = evaluator.find_optimal_threshold(
            self.models['LightGBM'], X_val, y_val, target_precision=0.65
        )
        evaluator.evaluate_model(self.models['LightGBM'], X_test, y_test, 'LightGBM', threshold=lgb_threshold)

        # CatBoost
        self.models['CatBoost'] = CatBoostClassifier(**cb_params)
        self.models['CatBoost'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30,
            verbose=False
        )

        cb_threshold = evaluator.find_optimal_threshold(
            self.models['CatBoost'], X_val, y_val, target_precision=0.65
        )
        evaluator.evaluate_model(self.models['CatBoost'], X_test, y_test, 'CatBoost', threshold=cb_threshold)

        # Plot comparison
        evaluator.plot_comparison(save_path=f'{self.signal_type.lower()}_models_comparison.png')

        # Select best model
        best_model_name = evaluator.get_best_model()
        self.best_model = self.models[best_model_name]
        self.optimal_threshold = evaluator.results[best_model_name]['threshold']

        # Analyze performance by market regime
        self.analyze_regime_performance(X_test, y_test, df_test)

        # Feature importance
        self.plot_feature_importance(self.best_model, X_train.columns, best_model_name)

        return {
            'results': evaluator.results,
            'best_model': best_model_name,
            'optimal_threshold': self.optimal_threshold,
            'best_params': {
                'XGBoost': xgb_params,
                'LightGBM': lgb_params,
                'CatBoost': cb_params
            }[best_model_name] if optimize_hyperparams else None
        }

    def analyze_regime_performance(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  df_test_original: pd.DataFrame):
        """
        Analyze model performance by market regime.

        Args:
            X_test: Test features
            y_test: Test target
            df_test_original: Original test data with market_regime
        """
        if self.best_model is None:
            logger.warning("No best model available for regime analysis")
            return

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        analysis_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'market_regime': df_test_original['market_regime'].values
        })

        logger.info(f"\n{'='*50}")
        logger.info(f"Performance by Market Regime - {self.signal_type}")
        logger.info(f"Using threshold: {self.optimal_threshold:.2f}")
        logger.info(f"{'='*50}")

        for regime in ['BULL', 'NEUTRAL', 'BEAR']:
            regime_data = analysis_df[analysis_df['market_regime'] == regime]
            if len(regime_data) > 0:
                regime_predictions = regime_data[regime_data['predicted'] == 1]

                if len(regime_predictions) > 0:
                    win_rate = regime_predictions['actual'].mean()
                    trades_taken = len(regime_predictions) / len(regime_data)

                    logger.info(f"\n{regime} Market:")
                    logger.info(f"  Total samples: {len(regime_data)}")
                    logger.info(f"  Trades taken: {len(regime_predictions)} ({trades_taken:.1%})")
                    logger.info(f"  Win rate on taken trades: {win_rate:.1%}")

                    expected_profit = win_rate * 3 - (1 - win_rate) * 3
                    logger.info(f"  Expected profit per trade: {expected_profit:.2%}")

                    if self.signal_type == 'BUY' and regime == 'BULL' and win_rate >= 0.60:
                        logger.info(f"  ✅ EXCELLENT - Trade {self.signal_type} in {regime}")
                    elif self.signal_type == 'SELL' and regime == 'BEAR' and win_rate >= 0.60:
                        logger.info(f"  ✅ EXCELLENT - Trade {self.signal_type} in {regime}")
                    elif win_rate >= 0.55:
                        logger.info(f"  ⚠️ ACCEPTABLE - Be selective with {self.signal_type} in {regime}")
                    else:
                        logger.info(f"  ❌ AVOID - Don't trade {self.signal_type} in {regime}")
                else:
                    logger.info(f"\n{regime} Market: No trades taken at threshold {self.optimal_threshold:.2f}")

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str, top_n: int = 20):
        """
        Plot feature importance.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = model.get_feature_importance()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Check for meta columns (shouldn't be any)
        meta_features = importance_df[importance_df['feature'].str.startswith('_meta_')]
        if not meta_features.empty:
            logger.error(f"❌ ERROR: Meta features found in importance: {meta_features['feature'].tolist()}")

        # Highlight regime features
        regime_features = importance_df[importance_df['feature'].str.contains('regime')]
        if not regime_features.empty:
            logger.info(f"\nMarket Regime Feature Importance:")
            for _, row in regime_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        plt.figure(figsize=(10, 8))

        colors = []
        for f in importance_df['feature']:
            if '_meta_' in f:
                colors.append('red')  # Should not happen
            elif 'regime' in f:
                colors.append('green')
            elif 'poc' in f.lower():
                colors.append('orange')
            else:
                colors.append('blue')

        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        plt.xlabel('Importance')
        plt.title(f'{self.signal_type} - {model_name} Feature Importance (Top {top_n})\n'
                 f'Green=Regime, Orange=POC, Blue=Other')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        save_path = f'{self.signal_type.lower()}_{model_name.lower()}_importance.png'
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.show()

    def save_models(self, path_prefix: str = 'models/'):
        """
        Save trained models and preprocessor.

        Args:
            path_prefix: Directory prefix for saving models
        """
        os.makedirs(path_prefix, exist_ok=True)

        # Save preprocessor
        preprocessor_path = f'{path_prefix}{self.signal_type.lower()}_preprocessor.pkl'
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Save all models with metadata
        for name, model in self.models.items():
            model_data = {
                'model': model,
                'threshold': self.optimal_threshold if name == self.best_model.__class__.__name__ else 0.5,
                'signal_type': self.signal_type,
                'timestamp': datetime.now().isoformat(),
                'view_version': 'ml_training_data_v2'  # Track which view version was used
            }
            model_path = f'{path_prefix}{self.signal_type.lower()}_{name.lower()}_model.pkl'
            joblib.dump(model_data, model_path)
            logger.info(f"{name} model saved to {model_path}")

        # Save best model separately
        if self.best_model:
            best_model_data = {
                'model': self.best_model,
                'threshold': self.optimal_threshold,
                'signal_type': self.signal_type,
                'timestamp': datetime.now().isoformat(),
                'view_version': 'ml_training_data_v2'
            }
            best_model_path = f'{path_prefix}{self.signal_type.lower()}_best_model.pkl'
            joblib.dump(best_model_data, best_model_path)
            logger.info(f"Best model saved to {best_model_path} with threshold {self.optimal_threshold:.2f}")


def main():
    """Main execution function."""
    import sys

    logger.info("="*60)
    logger.info("STARTING ML MODEL TRAINING WITH NEW VIEW STRUCTURE")
    logger.info("Using: fas.ml_training_data_v2")
    logger.info("="*60)

    # Check for quick test mode
    quick_test = '--quick' in sys.argv
    n_trials = 5 if quick_test else 30

    if quick_test:
        logger.info("Running in QUICK TEST mode with reduced trials")

    # Train BUY model
    logger.info("\n" + "="*40)
    logger.info("TRAINING BUY MODEL")
    logger.info("="*40)

    buy_pipeline = TradingModelPipeline(signal_type='BUY')
    buy_results = buy_pipeline.train_and_evaluate(optimize_hyperparams=True, n_trials=n_trials)
    buy_pipeline.save_models()

    # Train SELL model
    logger.info("\n" + "="*40)
    logger.info("TRAINING SELL MODEL")
    logger.info("="*40)

    sell_pipeline = TradingModelPipeline(signal_type='SELL')
    sell_results = sell_pipeline.train_and_evaluate(optimize_hyperparams=True, n_trials=n_trials)
    sell_pipeline.save_models()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*60)

    logger.info(f"\n📊 BUY Model Performance:")
    logger.info(f"Best Model: {buy_results['best_model']}")
    logger.info(f"Optimal Threshold: {buy_results['optimal_threshold']:.2f}")
    for metric, value in buy_results['results'][buy_results['best_model']].items():
        if metric in ['win_rate', 'precision', 'recall', 'accuracy', 'trades_taken']:
            logger.info(f"  {metric}: {value:.1%}")
        else:
            logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\n📊 SELL Model Performance:")
    logger.info(f"Best Model: {sell_results['best_model']}")
    logger.info(f"Optimal Threshold: {sell_results['optimal_threshold']:.2f}")
    for metric, value in sell_results['results'][sell_results['best_model']].items():
        if metric in ['win_rate', 'precision', 'recall', 'accuracy', 'trades_taken']:
            logger.info(f"  {metric}: {value:.1%}")
        else:
            logger.info(f"  {metric}: {value:.4f}")

    # Final recommendations
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS")
    logger.info("="*60)

    buy_metrics = buy_results['results'][buy_results['best_model']]
    sell_metrics = sell_results['results'][sell_results['best_model']]

    if buy_metrics['win_rate'] >= 0.60 and buy_metrics['trades_taken'] <= 0.35:
        logger.info("✅ BUY model ready for production with filtering strategy")
    else:
        logger.info("⚠️ BUY model needs further tuning")

    if sell_metrics['win_rate'] >= 0.60 and sell_metrics['trades_taken'] <= 0.35:
        logger.info("✅ SELL model ready for production with filtering strategy")
    else:
        logger.info("⚠️ SELL model needs further tuning")

    logger.info("\n💡 Remember: The goal is to FILTER signals, not predict all of them!")
    logger.info("   Target: 60-70% win rate on 20-30% of signals")


if __name__ == "__main__":
    main()