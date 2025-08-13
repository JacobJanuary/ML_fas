"""
ML Model Training Script for Trading Signals V2
================================================
Updated for new ml_training_data_v2 structure
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
        logging.FileHandler('ml_training_v2.log'),
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
        Load training data from new ml_training_data_v2 view.

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

        # Validate data quality
        if len(df) == 0:
            raise ValueError(f"No data found for {self.signal_type}")

        logger.info(f"Loaded {len(df)} records for {self.signal_type}")

        # Check win rate
        win_rate = df['target'].mean()
        logger.info(f"Base win rate: {win_rate:.1%}")

        if win_rate > 0.75:
            logger.warning(f"⚠️ Win rate {win_rate:.1%} seems too high!")
        elif win_rate < 0.25:
            logger.warning(f"⚠️ Win rate {win_rate:.1%} seems too low!")
        else:
            logger.info(f"✅ Win rate {win_rate:.1%} looks realistic")

        # Market regime distribution
        logger.info(f"Market regime distribution:")
        regime_stats = df.groupby('market_regime')['target'].agg(['count', 'mean'])
        for regime, stats in regime_stats.iterrows():
            logger.info(f"  {regime}: {stats['count']} signals, {stats['mean']:.1%} win rate")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # ========== MARKET REGIME FEATURES ==========
        # Create regime alignment features
        df['regime_signal_match'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BULL')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BEAR'))
        ).astype(float) * 10  # Weight importance

        df['regime_conflict'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BEAR')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BULL'))
        ).astype(float) * -10  # Negative weight

        df['regime_neutral'] = (df['market_regime'] == 'NEUTRAL').astype(float) * 5

        # Combination of regime and signal strength
        df['strong_signal_right_regime'] = (
            df['regime_signal_match'] *
            (df['strength_numeric'] >= 3).astype(float)
        )

        # Regime numeric encoding
        regime_weights = {'BULL': 1.0, 'NEUTRAL': 0.0, 'BEAR': -1.0}
        df['regime_numeric'] = df['market_regime'].map(regime_weights).fillna(0)

        # Interaction features
        df['regime_x_total_score'] = df['regime_numeric'] * df['total_score']
        df['regime_x_rsi'] = df['regime_numeric'] * df['rsi']
        df['regime_x_volume'] = df['regime_numeric'] * df['volume_zscore']

        # ========== PATTERN FEATURES ==========
        # Pattern intensity metrics (using existing pattern_count)
        df['has_multiple_patterns'] = (df['pattern_count'] > 1).astype(float)
        df['pattern_complexity'] = df['pattern_count'] * df['pattern_score']

        # Pattern confidence metrics
        pattern_conf_cols = ['pattern_1_confidence', 'pattern_2_confidence', 'pattern_3_confidence']
        existing_conf_cols = [col for col in pattern_conf_cols if col in df.columns]
        if existing_conf_cols:
            df['max_pattern_confidence'] = df[existing_conf_cols].max(axis=1)
            df['avg_pattern_confidence'] = df[existing_conf_cols].mean(axis=1)

        # ========== COMBINATION FEATURES ==========
        # Combo intensity metrics (using existing combo_count)
        df['has_combos'] = (df['combo_count'] > 0).astype(float)

        combo_score_cols = ['combo_1_score', 'combo_2_score']
        existing_score_cols = [col for col in combo_score_cols if col in df.columns]
        if existing_score_cols:
            df['max_combo_score'] = df[existing_score_cols].max(axis=1)

        # ========== SCORE RATIOS ==========
        df['pattern_to_indicator_ratio'] = df['pattern_score'] / (df['indicator_score'] + 1e-8)
        df['combo_to_total_ratio'] = df['combination_score'] / (df['total_score'] + 1e-8)

        # ========== VOLUME FEATURES ==========
        df['volume_pressure'] = df['buy_ratio_weighted'] * df['volume_zscore']
        df['cvd_momentum'] = df['cvd_delta'] / (abs(df['cvd_cumulative']) + 1e-8)

        # ========== VOLATILITY FEATURES ==========
        df['score_per_volatility'] = df['total_score'] / (df['atr_pct'] + 1e-8)

        # ========== TEMPORAL FEATURES ==========
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

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

        # Weight adjustments based on regime alignment
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
        unique_weights, counts = np.unique(weights, return_counts=True)
        for w, c in zip(unique_weights, counts):
            logger.info(f"  Weight {w}: {c} samples ({c/len(weights)*100:.1f}%)")

        return weights

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training with exclusion of meta fields.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fit encoders)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.engineer_features(df)

        # Separate target
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
            # Columns to exclude
            exclude_cols = [
                'target', 'timestamp', 'pair_symbol', 'id',
                'trading_pair_id', 'is_meme', 'signal_type', 'signal_strength'
            ] + categorical_cols

            # CRITICAL: Exclude all meta fields
            meta_cols = [col for col in df.columns if col.startswith('_meta_')]
            if meta_cols:
                logger.info(f"✅ Excluding {len(meta_cols)} meta columns: {meta_cols}")
                exclude_cols.extend(meta_cols)

            # Also exclude any suspicious columns
            dangerous_patterns = ['outcome', 'result', 'profit', 'loss', 'hours_to', 'exit']
            dangerous_cols = [col for col in df.columns
                            if any(pattern in col.lower() for pattern in dangerous_patterns)
                            and not col.startswith('_meta_')]

            if dangerous_cols:
                logger.warning(f"⚠️ Excluding potentially dangerous columns: {dangerous_cols}")
                exclude_cols.extend(dangerous_cols)

            # Form the final feature list
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]

            logger.info(f"Selected {len(self.feature_columns)} features for training")

            # Log sample of selected features
            logger.info(f"Sample features: {self.feature_columns[:10]}")

        # Select features
        X = df[self.feature_columns]

        # Handle missing values intelligently
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_count = X[col].isnull().sum()
            if null_count > 0:
                if null_count / len(X) < 0.5:  # Less than 50% missing
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                else:
                    # Too many missing values, fill with 0
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
        """Optimize XGBoost hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'gamma': trial.suggest_float('gamma', 0.1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
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

                # Penalize unrealistic scores
                if auc > 0.75:
                    auc = auc * 0.9

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
        """Optimize LightGBM hyperparameters."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.7),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),
                'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
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
                auc = auc * 0.9

            return auc

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"LightGBM best AUC: {study.best_value:.4f}")
        return study.best_params

    def optimize_catboost(self, n_trials: int = 30) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 128),
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
                auc = auc * 0.9

            return auc

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"CatBoost best AUC: {study.best_value:.4f}")
        return study.best_params


class ModelEvaluator:
    """Evaluate and compare model performance."""

    def __init__(self, signal_type: str):
        """Initialize evaluator."""
        self.signal_type = signal_type
        self.results = {}

    def evaluate_model(self, model: Any, X_test: pd.DataFrame,
                       y_test: pd.Series, model_name: str,
                       threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate single model performance."""
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

        # Calculate trading metrics
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

        # Realistic performance check
        if metrics['roc_auc'] > 0.75:
            logger.warning(f"⚠️ ROC-AUC {metrics['roc_auc']:.3f} seems high")

        if 0.60 <= metrics['win_rate'] <= 0.70 and 0.15 <= metrics['trades_taken'] <= 0.35:
            logger.info(f"✅ GOOD: {metrics['win_rate']:.1%} win rate on {metrics['trades_taken']:.1%} of signals")

        return metrics

    def find_optimal_threshold(self, model: Any, X_val: pd.DataFrame,
                              y_val: pd.Series, target_precision: float = 0.65) -> float:
        """Find optimal probability threshold."""
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

    def get_best_model(self) -> str:
        """Get the name of the best model."""
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
            logger.info(f"\nBest model: {best_model}")
            logger.info(f"Win rate: {self.results[best_model]['win_rate']:.1%}")
            logger.info(f"Trades taken: {self.results[best_model]['trades_taken']:.1%}")
        else:
            logger.warning("No model meets minimum criteria")
            best_model = list(self.results.keys())[0] if self.results else None

        return best_model


class TradingModelPipeline:
    """Main pipeline for training trading models."""

    def __init__(self, signal_type: str = 'BUY'):
        """Initialize pipeline."""
        self.signal_type = signal_type
        self.db = DatabaseConnection()
        self.preprocessor = DataPreprocessor(signal_type)
        self.models = {}
        self.best_model = None
        self.optimal_threshold = 0.5

    def train_and_evaluate(self, optimize_hyperparams: bool = True,
                          n_trials: int = 30) -> Dict[str, Any]:
        """Train and evaluate all models."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {self.signal_type} Models")
        logger.info(f"Using ml_training_data_v2")
        logger.info(f"{'='*50}\n")

        # Load and prepare data
        df = self.preprocessor.load_data(self.db)

        # Sort by timestamp for temporal split
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save original data
        df_original = df.copy()

        # Prepare features
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
        df_val = df_original.iloc[train_size:train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        df_test = df_original.iloc[train_size + val_size:]

        logger.info(f"Temporal split:")
        logger.info(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} win rate)")
        logger.info(f"  Val: {len(X_val)} samples ({y_val.mean():.1%} win rate)")
        logger.info(f"  Test: {len(X_test)} samples ({y_test.mean():.1%} win rate)")
        logger.info(f"  Train period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
        logger.info(f"  Test period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

        # Initialize evaluator
        evaluator = ModelEvaluator(self.signal_type)

        # Optimize or use default parameters
        if optimize_hyperparams:
            optimizer = ModelOptimizer(X_train, y_train, X_val, y_val, weights_train)

            logger.info("\nOptimizing XGBoost...")
            xgb_params = optimizer.optimize_xgboost(n_trials)

            logger.info("\nOptimizing LightGBM...")
            lgb_params = optimizer.optimize_lightgbm(n_trials)

            logger.info("\nOptimizing CatBoost...")
            cb_params = optimizer.optimize_catboost(min(n_trials, 20))
        else:
            # Conservative defaults
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

        # Train models
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

        # Select best model
        best_model_name = evaluator.get_best_model()
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.optimal_threshold = evaluator.results[best_model_name]['threshold']

        # Analyze performance by market regime
        self.analyze_regime_performance(X_test, y_test, df_test)

        return {
            'results': evaluator.results,
            'best_model': best_model_name,
            'optimal_threshold': self.optimal_threshold
        }

    def analyze_regime_performance(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  df_test: pd.DataFrame):
        """Analyze model performance by market regime."""
        if self.best_model is None:
            return

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        analysis_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'market_regime': df_test['market_regime'].values
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
                    logger.info(f"  Win rate: {win_rate:.1%}")

                    expected_profit = win_rate * 3 - (1 - win_rate) * 3
                    logger.info(f"  Expected profit: {expected_profit:.2%}")

                    # Recommendations
                    if self.signal_type == 'BUY' and regime == 'BULL' and win_rate >= 0.60:
                        logger.info(f"  ✅ EXCELLENT - Trade {self.signal_type} in {regime}")
                    elif self.signal_type == 'SELL' and regime == 'BEAR' and win_rate >= 0.60:
                        logger.info(f"  ✅ EXCELLENT - Trade {self.signal_type} in {regime}")
                    elif win_rate >= 0.55:
                        logger.info(f"  ⚠️ ACCEPTABLE - Be selective in {regime}")
                    else:
                        logger.info(f"  ❌ AVOID - Don't trade {self.signal_type} in {regime}")

    def save_models(self, path_prefix: str = 'models/'):
        """Save trained models and preprocessor."""
        os.makedirs(path_prefix, exist_ok=True)

        # Save preprocessor
        preprocessor_path = f'{path_prefix}{self.signal_type.lower()}_preprocessor_v2.pkl'
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Save models with metadata
        for name, model in self.models.items():
            model_data = {
                'model': model,
                'threshold': self.optimal_threshold if self.best_model == model else 0.5,
                'signal_type': self.signal_type,
                'timestamp': datetime.now().isoformat(),
                'view_version': 'ml_training_data_v2'
            }
            model_path = f'{path_prefix}{self.signal_type.lower()}_{name.lower()}_model_v2.pkl'
            joblib.dump(model_data, model_path)
            logger.info(f"{name} model saved to {model_path}")

        # Save best model
        if self.best_model:
            best_model_data = {
                'model': self.best_model,
                'threshold': self.optimal_threshold,
                'signal_type': self.signal_type,
                'timestamp': datetime.now().isoformat(),
                'view_version': 'ml_training_data_v2'
            }
            best_model_path = f'{path_prefix}{self.signal_type.lower()}_best_model_v2.pkl'
            joblib.dump(best_model_data, best_model_path)
            logger.info(f"Best model saved to {best_model_path} with threshold {self.optimal_threshold:.2f}")


def main():
    """Main execution function."""
    import sys

    logger.info("="*60)
    logger.info("ML MODEL TRAINING V2 - Using ml_training_data_v2")
    logger.info("="*60)

    quick_test = '--quick' in sys.argv
    n_trials = 5 if quick_test else 30

    if quick_test:
        logger.info("Running in QUICK TEST mode")

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
    if buy_results['best_model']:
        logger.info(f"Best Model: {buy_results['best_model']}")
        logger.info(f"Optimal Threshold: {buy_results['optimal_threshold']:.2f}")
        metrics = buy_results['results'][buy_results['best_model']]
        for metric, value in metrics.items():
            if metric in ['win_rate', 'precision', 'recall', 'accuracy', 'trades_taken']:
                logger.info(f"  {metric}: {value:.1%}")
            else:
                logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\n📊 SELL Model Performance:")
    if sell_results['best_model']:
        logger.info(f"Best Model: {sell_results['best_model']}")
        logger.info(f"Optimal Threshold: {sell_results['optimal_threshold']:.2f}")
        metrics = sell_results['results'][sell_results['best_model']]
        for metric, value in metrics.items():
            if metric in ['win_rate', 'precision', 'recall', 'accuracy', 'trades_taken']:
                logger.info(f"  {metric}: {value:.1%}")
            else:
                logger.info(f"  {metric}: {value:.4f}")

    logger.info("\n💡 Remember: The goal is to FILTER signals, not predict all of them!")
    logger.info("   Target: 60-70% win rate on 20-30% of signals")


if __name__ == "__main__":
    main()