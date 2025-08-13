"""
ML Model Training Script for Trading Signals
=============================================
Production-ready script for training and optimizing ML models
for BUY and SELL trading signals prediction.
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
        Load training data from database.

        Args:
            db: Database connection instance

        Returns:
            DataFrame with training data
        """
        query = f"""
        SELECT 
            -- Target
            target_with_stoploss,
            
            -- Signal type (нужен для создания весов)
            signal_type,
            
            -- Core features
            indicator_score, pattern_score, combination_score, total_score,
            strength_numeric, 
            
            -- Market features
            close_price, price_change_pct, atr_pct, rsi_zone,
            
            -- MARKET REGIME - CRITICAL FEATURE
            market_regime,
            
            -- Volume features
            buy_ratio, buy_ratio_weighted, normalized_imbalance, 
            smoothed_imbalance, volume_zscore, cvd_delta, cvd_cumulative,
            
            -- Future/Funding features
            oi_delta_pct, funding_rate_avg,
            
            -- Technical indicators
            rsi, rs_value, rs_momentum, atr,
            macd_line, macd_signal, macd_histogram,
            
            -- POC levels
            poc_24h, poc_7d, poc_30d,
            poc_volume_24h, poc_volume_7d,
            
            -- Pattern features (impacts and confidences)
            pattern_1_impact, pattern_1_confidence,
            pattern_2_impact, pattern_2_confidence,
            pattern_3_impact, pattern_3_confidence,
            pattern_4_impact, pattern_4_confidence,
            pattern_5_impact, pattern_5_confidence,
            
            -- Pattern binary features
            has_distribution, has_accumulation, has_volume_anomaly,
            has_momentum_exhaustion, has_oi_explosion, 
            has_squeeze_ignition, has_cvd_divergence,
            
            -- Combination features (scores and confidences)
            combo_1_score, combo_1_confidence,
            combo_2_score, combo_2_confidence,
            combo_3_score, combo_3_confidence,
            
            -- Combination binary features
            has_volume_distribution, has_volume_accumulation,
            has_institutional_surge, has_squeeze_momentum,
            has_smart_accumulation,
            
            -- Categorical features
            is_meme,
            pattern_1_name, pattern_2_name, pattern_3_name,
            combo_1_name, combo_2_name,
            
            -- Meta features (БЕЗ данных из будущего!)
            timestamp, pair_symbol,
            
            -- Signal strength (text)
            signal_strength
            
        FROM fas.ml_new_training_data
        WHERE signal_type = '{self.signal_type}'
            AND target_with_stoploss IS NOT NULL
            AND timestamp < NOW() - INTERVAL '60 hours'
        ORDER BY timestamp
        """

        logger.info(f"Loading {self.signal_type} data from database...")
        df = db.fetch_data(query)
        logger.info(f"Loaded {len(df)} records for {self.signal_type}")
        logger.info(f"Market regime distribution:")
        logger.info(f"{df['market_regime'].value_counts()}")

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

        # ========== MARKET REGIME FEATURES (HIGH IMPORTANCE) ==========
        # Создаем признаки соответствия режиму рынка с увеличенным весом
        df['regime_signal_match'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BULL')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BEAR'))
        ).astype(float) * 10  # Умножаем на 10 для увеличения важности

        df['regime_conflict'] = (
            ((df['signal_type'] == 'BUY') & (df['market_regime'] == 'BEAR')) |
            ((df['signal_type'] == 'SELL') & (df['market_regime'] == 'BULL'))
        ).astype(float) * -10  # Негативный вес для конфликта

        df['regime_neutral'] = (df['market_regime'] == 'NEUTRAL').astype(float) * 5

        # Комбинированные признаки режима и силы сигнала
        df['strong_signal_right_regime'] = (
            df['regime_signal_match'] *
            (df['signal_strength'].isin(['STRONG', 'VERY_STRONG'])).astype(float)
        )

        df['weak_signal_wrong_regime'] = (
            df['regime_conflict'] *
            (df['signal_strength'].isin(['WEAK', 'MODERATE'])).astype(float)
        )

        # Числовое кодирование режима для лучшего обучения
        regime_weights = {'BULL': 1.0, 'NEUTRAL': 0.0, 'BEAR': -1.0}
        df['regime_numeric'] = df['market_regime'].map(regime_weights)

        # Взаимодействие режима с ключевыми индикаторами
        df['regime_x_total_score'] = df['regime_numeric'] * df['total_score']
        df['regime_x_rsi'] = df['regime_numeric'] * df['rsi']
        df['regime_x_volume'] = df['regime_numeric'] * df['volume_zscore']

        # ========== ORIGINAL FEATURES ==========
        # Price-related features
        df['price_to_poc_24h'] = (df['close_price'] - df['poc_24h']) / df['poc_24h'] * 100
        df['price_to_poc_7d'] = (df['close_price'] - df['poc_7d']) / df['poc_7d'] * 100

        # Pattern intensity
        df['total_pattern_impact'] = df[[col for col in df.columns if 'pattern' in col and 'impact' in col]].sum(axis=1)
        df['avg_pattern_confidence'] = df[[col for col in df.columns if 'pattern' in col and 'confidence' in col]].mean(axis=1)

        # Combination intensity
        df['total_combo_score'] = df[[col for col in df.columns if 'combo' in col and 'score' in col]].sum(axis=1)
        df['avg_combo_confidence'] = df[[col for col in df.columns if 'combo' in col and 'confidence' in col]].mean(axis=1)

        # Score ratios
        df['pattern_to_indicator_ratio'] = df['pattern_score'] / (df['indicator_score'] + 1e-8)
        df['combo_to_total_ratio'] = df['combination_score'] / (df['total_score'] + 1e-8)

        # Volume features
        df['volume_pressure'] = df['buy_ratio_weighted'] * df['volume_zscore']
        df['cvd_momentum'] = df['cvd_delta'] / (abs(df['cvd_cumulative']) + 1e-8)

        # Volatility-adjusted scores
        df['score_per_volatility'] = df['total_score'] / (df['atr_pct'] + 1e-8)

        # ========== TREND STRENGTH FEATURES ==========
        # Добавляем признаки силы тренда
        df['bull_market_strength'] = (df['market_regime'] == 'BULL').astype(float) * abs(df['total_score'])
        df['bear_market_strength'] = (df['market_regime'] == 'BEAR').astype(float) * abs(df['total_score'])

        # Hour of day (cyclical encoding)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (cyclical encoding)
        df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

        logger.info(f"Engineered {len([col for col in df.columns if 'regime' in col])} market regime features")

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

        # Увеличиваем вес для правильных комбинаций режим-сигнал
        # BUY в BULL market - отличная комбинация
        bull_buy_mask = (df['signal_type'] == 'BUY') & (df['market_regime'] == 'BULL')
        weights[bull_buy_mask] = 1.5

        # SELL в BEAR market - отличная комбинация
        bear_sell_mask = (df['signal_type'] == 'SELL') & (df['market_regime'] == 'BEAR')
        weights[bear_sell_mask] = 1.5

        # Уменьшаем вес для конфликтных комбинаций
        # BUY в BEAR market - плохая комбинация
        bear_buy_mask = (df['signal_type'] == 'BUY') & (df['market_regime'] == 'BEAR')
        weights[bear_buy_mask] = 0.5

        # SELL в BULL market - плохая комбинация
        bull_sell_mask = (df['signal_type'] == 'SELL') & (df['market_regime'] == 'BULL')
        weights[bull_sell_mask] = 0.5

        # Дополнительно увеличиваем вес для успешных сделок в правильном режиме
        # Это поможет модели лучше выучить паттерны успеха
        successful_trades = y == 1
        weights[bull_buy_mask & successful_trades] = 2.0
        weights[bear_sell_mask & successful_trades] = 2.0

        # Логируем распределение весов
        logger.info(f"Sample weights distribution:")
        logger.info(f"  Weight 2.0 (best): {(weights == 2.0).sum()} samples")
        logger.info(f"  Weight 1.5 (good): {(weights == 1.5).sum()} samples")
        logger.info(f"  Weight 1.0 (neutral): {(weights == 1.0).sum()} samples")
        logger.info(f"  Weight 0.5 (bad): {(weights == 0.5).sum()} samples")

        return weights

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fit encoders)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.engineer_features(df)

        # Separate target
        y = df['target_with_stoploss'].astype(int)

        # Handle categorical features
        categorical_cols = ['market_regime', 'pattern_1_name', 'pattern_2_name',
                          'pattern_3_name', 'combo_1_name', 'combo_2_name']

        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    # Fit and transform
                    le = LabelEncoder()
                    # Handle NaN values
                    df[col] = df[col].fillna('unknown')
                    df[f'{col}_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    # Transform only
                    df[col] = df[col].fillna('unknown')
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )

        # Select feature columns
        if is_training:
            # Define feature columns to use
            # ВАЖНО: Исключаем поля с данными из будущего!
            exclude_cols = [
                'target_with_stoploss', 'timestamp', 'pair_symbol',
                'max_profit_pct', 'max_loss_pct', 'hours_to_outcome',  # ДАННЫЕ ИЗ БУДУЩЕГО
                'outcome', 'outcome_timestamp',  # ДАННЫЕ ИЗ БУДУЩЕГО
                'is_meme', 'signal_type', 'signal_strength'  # Служебные поля
            ] + categorical_cols

            self.feature_columns = [col for col in df.columns if col not in exclude_cols]

            # Проверка на случайно попавшие опасные поля
            dangerous_cols = [col for col in self.feature_columns if any(
                word in col.lower() for word in ['outcome', 'result', 'profit', 'loss', 'hours_to']
            )]
            if dangerous_cols:
                logger.warning(f"⚠️ WARNING: Potentially dangerous columns found: {dangerous_cols}")
                self.feature_columns = [col for col in self.feature_columns if col not in dangerous_cols]

        # Select features
        X = df[self.feature_columns].fillna(0)

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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 50,  # For XGBoost 2.0+
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
                return roc_auc_score(self.y_val, y_pred)
            except Exception as e:
                logger.warning(f"Trial failed with error: {e}")
                return 0.5  # Return baseline score on error

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"XGBoost best AUC: {study.best_value:.4f}")

        # Return best params without early_stopping_rounds (it's a fit param)
        best_params = study.best_params.copy()
        best_params['early_stopping_rounds'] = 50
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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 2),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 2),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
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
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            y_pred = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, y_pred)

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
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
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
                early_stopping_rounds=50,
                verbose=False
            )

            y_pred = model.predict_proba(self.X_val)[:, 1]
            return roc_auc_score(self.y_val, y_pred)

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
                       y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluate single model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean()
        }

        # Calculate expected profit (simplified)
        # Assuming 3% profit on win, -3% loss on loss
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()

        if (tp + fp) > 0:
            metrics['expected_profit'] = (tp * 3 - fp * 3) / (tp + fp)
            metrics['win_rate'] = tp / (tp + fp)
        else:
            metrics['expected_profit'] = 0
            metrics['win_rate'] = 0

        self.results[model_name] = metrics

        logger.info(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # ПРОВЕРКА НА РЕАЛИСТИЧНОСТЬ
        if metrics['win_rate'] > 0.7:
            logger.warning(f"⚠️ WARNING: Win rate {metrics['win_rate']:.1%} seems too high!")
            logger.warning("  This may indicate data leakage or overfitting")
            logger.warning("  Expected win rate for crypto trading: 40-60%")

        if metrics['roc_auc'] > 0.9:
            logger.warning(f"⚠️ WARNING: ROC-AUC {metrics['roc_auc']:.3f} is suspiciously high!")
            logger.warning("  Check for data leakage in features")

        return metrics

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

        metrics = ['roc_auc', 'precision', 'recall', 'f1', 'expected_profit', 'win_rate']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            models = list(self.results.keys())
            values = [self.results[m][metric] for m in models]

            bars = ax.bar(models, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1] if metric != 'expected_profit' else [min(values) - 0.5, max(values) + 0.5])

            # Color bars based on performance
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()

    def get_best_model(self) -> str:
        """
        Get the name of the best model based on expected profit.

        Returns:
            Name of the best model
        """
        if not self.results:
            return None

        best_model = max(self.results.items(),
                        key=lambda x: x[1]['expected_profit'])[0]

        logger.info(f"\nBest model for {self.signal_type}: {best_model}")
        logger.info(f"Expected profit: {self.results[best_model]['expected_profit']:.4f}")

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

        # ВАЖНО: Сортируем по времени для корректного разделения
        df = df.sort_values('timestamp')

        # Сохраняем исходные данные до препроцессинга для создания весов
        df_original = df.copy()

        X, y = self.preprocessor.prepare_features(df, is_training=True)

        # Создаем веса для семплов на основе market regime
        sample_weights = self.preprocessor.create_sample_weights(df_original, y)

        # ВРЕМЕННОЕ РАЗДЕЛЕНИЕ (не случайное!) чтобы избежать утечки
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        # Разделяем по времени
        X_train = X[:train_size]
        y_train = y[:train_size]
        weights_train = sample_weights[:train_size]
        df_train = df_original[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        weights_val = sample_weights[train_size:train_size + val_size]
        df_val = df_original[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        weights_test = sample_weights[train_size + val_size:]
        df_test_original = df_original[train_size + val_size:]

        logger.info(f"Data split (temporal):")
        logger.info(f"  Train: {len(X_train)} samples (first 70%)")
        logger.info(f"  Val: {len(X_val)} samples (next 15%)")
        logger.info(f"  Test: {len(X_test)} samples (last 15%)")
        logger.info(f"  Train period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
        logger.info(f"  Val period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
        logger.info(f"  Test period: {df_test_original['timestamp'].min()} to {df_test_original['timestamp'].max()}")

        # Разделяем оригинальные данные для анализа по режимам
        df_temp, df_test_original = train_test_split(
            df_original, test_size=0.15, random_state=42, stratify=y
        )

        logger.info(f"Target distribution:")
        logger.info(f"  Train: {y_train.mean():.2%} win rate")
        logger.info(f"  Val: {y_val.mean():.2%} win rate")
        logger.info(f"  Test: {y_test.mean():.2%} win rate")

        if 'market_regime' in df_test_original.columns:
            logger.info(f"Market regime in test set:")
            logger.info(f"{df_test_original['market_regime'].value_counts()}")

        # Initialize evaluator
        evaluator = ModelEvaluator(self.signal_type)

        if optimize_hyperparams:
            # Optimize hyperparameters with sample weights
            optimizer = ModelOptimizer(X_train, y_train, X_val, y_val, weights_train)

            logger.info("\nOptimizing XGBoost...")
            xgb_params = optimizer.optimize_xgboost(n_trials)

            logger.info("\nOptimizing LightGBM...")
            lgb_params = optimizer.optimize_lightgbm(n_trials)

            logger.info("\nOptimizing CatBoost...")
            cb_params = optimizer.optimize_catboost(min(n_trials, 30))  # CatBoost slower, limit trials
        else:
            # Use default parameters
            xgb_params = {
                'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                'early_stopping_rounds': 50, 'enable_categorical': False
            }
            lgb_params = {
                'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.1,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'random_state': 42,
                'verbose': -1
            }
            cb_params = {
                'iterations': 300, 'depth': 6, 'learning_rate': 0.1,
                'random_state': 42, 'verbose': False
            }

        # Train final models
        logger.info("\nTraining final models with sample weights...")

        # XGBoost
        # Remove early_stopping_rounds from params if it exists (it's now part of constructor)
        xgb_train_params = xgb_params.copy()
        if 'early_stopping_rounds' in xgb_train_params:
            xgb_train_params['early_stopping_rounds'] = 50
        self.models['XGBoost'] = xgb.XGBClassifier(**xgb_train_params)
        self.models['XGBoost'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        evaluator.evaluate_model(self.models['XGBoost'], X_test, y_test, 'XGBoost')

        # LightGBM
        self.models['LightGBM'] = lgb.LGBMClassifier(**lgb_params, verbose=-1)
        self.models['LightGBM'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        evaluator.evaluate_model(self.models['LightGBM'], X_test, y_test, 'LightGBM')

        # CatBoost
        cb_train_params = cb_params.copy()
        cb_train_params['verbose'] = False
        self.models['CatBoost'] = CatBoostClassifier(**cb_train_params)
        self.models['CatBoost'].fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        evaluator.evaluate_model(self.models['CatBoost'], X_test, y_test, 'CatBoost')

        # Plot comparison
        evaluator.plot_comparison(save_path=f'{self.signal_type.lower()}_models_comparison.png')

        # Select best model
        best_model_name = evaluator.get_best_model()
        self.best_model = self.models[best_model_name]

        # Analyze performance by market regime
        self.analyze_regime_performance(X_test, y_test, df_test_original)

        # Feature importance for best model
        self.plot_feature_importance(self.best_model, X_train.columns, best_model_name)

        return {
            'results': evaluator.results,
            'best_model': best_model_name,
            'best_params': {
                'XGBoost': xgb_params,
                'LightGBM': lgb_params,
                'CatBoost': cb_params
            }[best_model_name] if optimize_hyperparams else None
        }

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

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Highlight regime-related features
        regime_features = importance_df[importance_df['feature'].str.contains('regime')]
        if not regime_features.empty:
            logger.info(f"\nMarket Regime Feature Importance:")
            for _, row in regime_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Plot
        plt.figure(figsize=(10, 8))

        # Color bars differently for regime features
        colors = ['red' if 'regime' in f else 'blue' for f in importance_df['feature']]

        plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        plt.xlabel('Importance')
        plt.title(f'{self.signal_type} - {model_name} Feature Importance (Top {top_n})\n'
                 f'Red = Market Regime Features, Blue = Other Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        save_path = f'{self.signal_type.lower()}_{model_name.lower()}_importance.png'
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.show()

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

        # Get predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba,
            'market_regime': df_test_original['market_regime'].values[:len(y_test)]
        })

        # Calculate metrics by regime
        logger.info(f"\n{'='*50}")
        logger.info(f"Performance by Market Regime - {self.signal_type}")
        logger.info(f"{'='*50}")

        for regime in ['BULL', 'NEUTRAL', 'BEAR']:
            regime_data = analysis_df[analysis_df['market_regime'] == regime]
            if len(regime_data) > 0:
                accuracy = (regime_data['predicted'] == regime_data['actual']).mean()
                win_rate = regime_data['actual'].mean()
                predicted_win_rate = regime_data['predicted'].mean()

                logger.info(f"\n{regime} Market:")
                logger.info(f"  Samples: {len(regime_data)}")
                logger.info(f"  Actual Win Rate: {win_rate:.2%}")
                logger.info(f"  Predicted Win Rate: {predicted_win_rate:.2%}")
                logger.info(f"  Accuracy: {accuracy:.2%}")

                # Expected profit
                tp = ((regime_data['predicted'] == 1) & (regime_data['actual'] == 1)).sum()
                fp = ((regime_data['predicted'] == 1) & (regime_data['actual'] == 0)).sum()
                if (tp + fp) > 0:
                    expected_profit = (tp * 3 - fp * 3) / (tp + fp)
                    logger.info(f"  Expected Profit: {expected_profit:.2%}")

                # Recommendation
                if self.signal_type == 'BUY' and regime == 'BULL':
                    logger.info(f"  ✅ RECOMMENDED - Trade this signal in {regime}")
                elif self.signal_type == 'SELL' and regime == 'BEAR':
                    logger.info(f"  ✅ RECOMMENDED - Trade this signal in {regime}")
                elif regime == 'NEUTRAL':
                    logger.info(f"  ⚠️ CAUTION - Be selective in {regime}")
                else:
                    logger.info(f"  ❌ AVOID - Don't trade {self.signal_type} in {regime}")

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

        # Save all models
        for name, model in self.models.items():
            model_path = f'{path_prefix}{self.signal_type.lower()}_{name.lower()}_model.pkl'
            joblib.dump(model, model_path)
            logger.info(f"{name} model saved to {model_path}")

        # Save best model separately
        if self.best_model:
            best_model_path = f'{path_prefix}{self.signal_type.lower()}_best_model.pkl'
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")


def main():
    """Main execution function."""
    import sys

    # Check for quick test mode
    quick_test = '--quick' in sys.argv
    n_trials = 10 if quick_test else 50

    if quick_test:
        logger.info("Running in QUICK TEST mode with reduced trials")

    # Train BUY model
    buy_pipeline = TradingModelPipeline(signal_type='BUY')
    buy_results = buy_pipeline.train_and_evaluate(optimize_hyperparams=True, n_trials=n_trials)
    buy_pipeline.save_models()

    # Train SELL model
    sell_pipeline = TradingModelPipeline(signal_type='SELL')
    sell_results = sell_pipeline.train_and_evaluate(optimize_hyperparams=True, n_trials=n_trials)
    sell_pipeline.save_models()

    # Summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)

    logger.info(f"\nBUY Model Performance:")
    logger.info(f"Best Model: {buy_results['best_model']}")
    for metric, value in buy_results['results'][buy_results['best_model']].items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\nSELL Model Performance:")
    logger.info(f"Best Model: {sell_results['best_model']}")
    for metric, value in sell_results['results'][sell_results['best_model']].items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()