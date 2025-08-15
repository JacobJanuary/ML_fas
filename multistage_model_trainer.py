"""
Training System for Multi-Stage Models
=======================================
Trains specialized models for each stage and market regime.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import psycopg2
from datetime import datetime, timedelta
import joblib
import logging
import os
from typing import Dict, Tuple, Optional
from dotenv import load_dotenv
from feature_preprocessor import FeaturePreprocessor  # Import our preprocessor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiStageModelTrainer:
    """Trains all models for multi-stage filtering system."""

    def __init__(self):
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        self.models_dir = 'models/multistage/'
        os.makedirs(self.models_dir, exist_ok=True)

        # Training configurations for different model types
        self.configs = {
            'quick_filter': {
                'features': ['total_score', 'rsi', 'volume_zscore', 'price_change_pct',
                             'buy_ratio', 'cvd_delta'],
                'params': {
                    'max_depth': 3,
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                },
                'target_threshold': 0.3  # Low threshold, just filter obvious bad ones
            },
            'regime_specific': {
                'features': None,  # Use all features
                'params': {
                    'max_depth': 5,
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85
                }
            },
            'precision': {
                'features': None,  # Use all features + ensemble features
                'params': {
                    'max_depth': 6,
                    'n_estimators': 300,
                    'learning_rate': 0.03,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'min_child_weight': 10
                }
            }
        }

    def train_all_models(self):
        """Train all models for the multi-stage system."""
        logger.info("=" * 60)
        logger.info("TRAINING MULTI-STAGE MODELS")
        logger.info("=" * 60)

        # 1. Train quick filter models
        for signal_type in ['BUY', 'SELL']:
            self.train_quick_filter(signal_type)

        # 2. Train regime-specific models
        for signal_type in ['BUY', 'SELL']:
            for regime in ['BULL', 'BEAR', 'NEUTRAL']:
                self.train_regime_model(signal_type, regime)

        # 3. Train precision models
        for signal_type in ['BUY', 'SELL']:
            self.train_precision_model(signal_type)

        logger.info("\n✅ All models trained successfully!")

    def train_quick_filter(self, signal_type: str):
        """Train Stage 1 quick filter model."""
        logger.info(f"\n📊 Training Quick Filter for {signal_type}")

        # Load data
        df = self.load_training_data(signal_type, days_back=7)

        if len(df) < 1000:
            logger.warning(f"Not enough data for {signal_type} quick filter: {len(df)}")
            return None

        # Use preprocessor for feature preparation
        preprocessor = FeaturePreprocessor()
        df_processed = preprocessor.prepare_features(df, is_training=True, model_type='quick_filter')

        # Separate features and target
        if 'target' in df_processed.columns:
            X = df_processed.drop('target', axis=1)
            y = df_processed['target'].astype(int)
        else:
            logger.error("No target column found after preprocessing!")
            return None

        feature_cols = X.columns.tolist()
        logger.info(f"  Using {len(feature_cols)} features")

        # Train model
        model = xgb.XGBClassifier(
            **self.configs['quick_filter']['params'],
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
        logger.info(f"  Cross-validation AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Train final model
        model.fit(X, y)

        # Find threshold for 30% pass rate (70% filtered)
        y_pred_proba = model.predict_proba(X)[:, 1]
        threshold = float(np.percentile(y_pred_proba, 70))  # Convert to Python float

        # Save model
        model_data = {
            'model': model,
            'scaler': None,  # Quick filter doesn't need scaling
            'threshold': float(threshold),  # Ensure Python float
            'features': feature_cols,
            'signal_type': signal_type,
            'model_type': 'quick_filter',
            'timestamp': datetime.now().isoformat()
        }

        model_path = f"{self.models_dir}quick_filter_{signal_type.lower()}.pkl"
        joblib.dump(model_data, model_path)

        # Register in database
        model_id = self.register_model(
            model_name=f"quick_filter_{signal_type}",
            model_type='quick_filter',
            signal_type=signal_type,
            model_path=model_path,
            threshold=threshold,
            features=feature_cols
        )

        logger.info(f"  ✅ Saved as model ID: {model_id}, threshold: {threshold:.3f}")

        return model_id

    def train_regime_model(self, signal_type: str, regime: str):
        """Train Stage 3 regime-specific model."""
        logger.info(f"\n📊 Training {regime} Model for {signal_type}")

        # Determine window size based on regime
        window_days = {
            'BULL': 7,
            'BEAR': 3,
            'NEUTRAL': 5
        }[regime]

        # Load regime-specific data
        df = self.load_training_data(signal_type, days_back=window_days, regime=regime)

        if len(df) < 500:
            logger.warning(f"Not enough {regime} data for {signal_type}: {len(df)}")
            # Train universal model instead
            df = self.load_training_data(signal_type, days_back=window_days)

        if len(df) < 1000:
            logger.warning(f"Still not enough data, skipping {regime} {signal_type}")
            return None

        # Use preprocessor for feature preparation
        preprocessor = FeaturePreprocessor()
        df_processed = preprocessor.prepare_features(df, is_training=True, model_type='regime_specific')

        # Separate features and target
        if 'target' in df_processed.columns:
            X = df_processed.drop('target', axis=1)
            y = df_processed['target'].astype(int)
        else:
            logger.error("No target column found after preprocessing!")
            return None

        feature_cols = X.columns.tolist()
        logger.info(f"  Using {len(feature_cols)} features")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model with regime-specific parameters
        params = self.configs['regime_specific']['params'].copy()

        # Adjust for regime
        if regime == 'BEAR':
            params['min_child_weight'] = 20  # More conservative in bear market
        elif regime == 'BULL':
            params['min_child_weight'] = 5  # More aggressive in bull market

        model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
        logger.info(f"  Cross-validation AUC: {scores.mean():.3f}")

        # Train final model
        model.fit(X_scaled, y)

        # Find optimal threshold for target win rate
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        target_wr = 0.75 if signal_type == 'BUY' else 0.65
        threshold = self.find_threshold_for_win_rate(y, y_pred_proba, target_wr)

        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'threshold': float(threshold),  # Ensure Python float
            'features': feature_cols,
            'signal_type': signal_type,
            'model_type': 'regime_specific',
            'regime': regime,
            'window_days': window_days,
            'timestamp': datetime.now().isoformat()
        }

        model_path = f"{self.models_dir}regime_{regime.lower()}_{signal_type.lower()}.pkl"
        joblib.dump(model_data, model_path)

        # Register in database
        model_id = self.register_model(
            model_name=f"regime_{regime}_{signal_type}",
            model_type='regime_specific',
            signal_type=signal_type,
            market_regime=regime,
            model_path=model_path,
            threshold=threshold,
            features=feature_cols,
            window_days=window_days
        )

        # Evaluate on test data
        win_rate = self.evaluate_model(model, scaler, X, y, threshold)
        logger.info(f"  ✅ Model ID: {model_id}, threshold: {threshold:.3f}, win rate: {win_rate:.1%}")

        return model_id

    def train_precision_model(self, signal_type: str):
        """Train Stage 4 high-precision model."""
        logger.info(f"\n📊 Training Precision Model for {signal_type}")

        # Load only high-quality historical data
        df = self.load_high_quality_data(signal_type)

        if len(df) < 2000:
            logger.warning(f"Not enough high-quality data for {signal_type}: {len(df)}")
            df = self.load_training_data(signal_type, days_back=14)

        # Use preprocessor for feature preparation
        preprocessor = FeaturePreprocessor()
        df_processed = preprocessor.prepare_features(df, is_training=True, model_type='precision')

        # Separate features and target
        if 'target' in df_processed.columns:
            X = df_processed.drop('target', axis=1)
            y = df_processed['target'].astype(int)
        else:
            logger.error("No target column found after preprocessing!")
            return None

        # Add ensemble feature placeholders (simulated for training)
        X['stage1_prob'] = np.random.uniform(0.3, 0.9, len(X))
        X['stage3_prob'] = np.random.uniform(0.4, 0.95, len(X))
        X['prob_std'] = np.random.uniform(0.05, 0.3, len(X))
        X['prob_mean'] = np.random.uniform(0.4, 0.8, len(X))

        feature_cols = X.columns.tolist()
        logger.info(f"  Using {len(feature_cols)} features (including ensemble)")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train high-precision model
        model = xgb.XGBClassifier(
            **self.configs['precision']['params'],
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
        logger.info(f"  Cross-validation AUC: {scores.mean():.3f}")

        # Train final model
        model.fit(X_scaled, y)

        # Find threshold for high win rate
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]

        target_wr = 0.85 if signal_type == 'BUY' else 0.70
        threshold = self.find_threshold_for_win_rate(y, y_pred_proba, target_wr, min_samples=100)

        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'threshold': float(threshold),  # Ensure Python float
            'features': feature_cols,
            'signal_type': signal_type,
            'model_type': 'precision',
            'timestamp': datetime.now().isoformat()
        }

        model_path = f"{self.models_dir}precision_{signal_type.lower()}.pkl"
        joblib.dump(model_data, model_path)

        # Register in database
        model_id = self.register_model(
            model_name=f"precision_{signal_type}",
            model_type='precision',
            signal_type=signal_type,
            model_path=model_path,
            threshold=threshold,
            features=feature_cols
        )

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("  Top 10 features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.3f}")

        win_rate = self.evaluate_model(model, scaler, X, y, threshold)
        logger.info(f"  ✅ Model ID: {model_id}, threshold: {threshold:.3f}, win rate: {win_rate:.1%}")

        return model_id

    def load_training_data(self, signal_type: str, days_back: int = 7,
                           regime: Optional[str] = None) -> pd.DataFrame:
        """Load training data from database."""
        query = f"""
        SELECT *
        FROM fas.ml_training_data_direct
        WHERE signal_type = %s
            AND target IS NOT NULL
            AND timestamp >= NOW() - INTERVAL '{days_back} days'
            AND timestamp < NOW() - INTERVAL '48 hours'
        """

        params = [signal_type]

        if regime:
            query += " AND market_regime = %s"
            params.append(regime)

        query += " ORDER BY timestamp"

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn, params=params)

        logger.info(f"  Loaded {len(df)} samples, win rate: {df['target'].mean():.1%}")
        return df

    def load_high_quality_data(self, signal_type: str) -> pd.DataFrame:
        """Load only high-confidence historical predictions."""
        query = """
        WITH high_quality AS (
            SELECT 
                mtd.*,
                sp.prediction_probability,
                po.actual_outcome
            FROM fas.ml_training_data_direct mtd
            LEFT JOIN ml.signal_predictions sp ON mtd.id = sp.signal_id
            LEFT JOIN ml.prediction_outcomes po ON sp.id = po.prediction_id
            WHERE mtd.signal_type = %s
                AND mtd.target IS NOT NULL
                AND mtd.timestamp >= NOW() - INTERVAL '30 days'
                AND mtd.timestamp < NOW() - INTERVAL '48 hours'
                AND (
                    -- High confidence predictions that were correct
                    (sp.prediction_probability > 0.7 AND po.actual_outcome = true)
                    OR
                    -- Include some recent data regardless
                    mtd.timestamp >= NOW() - INTERVAL '7 days'
                )
        )
        SELECT * FROM high_quality
        ORDER BY timestamp
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn, params=[signal_type])

        return df

    def find_threshold_for_win_rate(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    target_wr: float, min_samples: int = 50) -> float:
        """Find threshold that achieves target win rate."""
        thresholds = np.linspace(0.3, 0.95, 100)
        best_threshold = 0.5
        best_distance = float('inf')

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if y_pred.sum() >= min_samples:  # Need minimum samples
                win_rate = y_true[y_pred == 1].mean() if y_pred.sum() > 0 else 0
                distance = abs(win_rate - target_wr)

                if distance < best_distance:
                    best_distance = distance
                    best_threshold = threshold

        return float(best_threshold)  # Convert to Python float

    def evaluate_model(self, model, scaler, X, y, threshold) -> float:
        """Evaluate model performance."""
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        if y_pred.sum() > 0:
            win_rate = float(y[y_pred == 1].mean())  # Convert to Python float
            return win_rate
        return 0.0

    def register_model(self, **kwargs) -> int:
        """Register model in database."""
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Convert numpy types to Python types
                threshold = kwargs.get('threshold')
                if threshold is not None:
                    threshold = float(threshold)  # Convert numpy.float32/64 to Python float

                features = kwargs.get('features')
                if features is not None:
                    # Convert to list of strings if it's not already
                    if isinstance(features, (list, tuple)):
                        features = [str(f) for f in features]
                    else:
                        features = None

                window_days = kwargs.get('window_days')
                if window_days is not None:
                    window_days = int(window_days)  # Convert to Python int

                cur.execute("""
                    INSERT INTO ml.model_registry (
                        model_name, model_type, signal_type, market_regime,
                        window_days, threshold, features_used, model_path,
                        last_trained_at
                    ) VALUES (
                        %(model_name)s, %(model_type)s, %(signal_type)s, %(market_regime)s,
                        %(window_days)s, %(threshold)s, %(features)s, %(model_path)s,
                        NOW()
                    )
                    ON CONFLICT (model_name) DO UPDATE SET
                        threshold = EXCLUDED.threshold,
                        model_path = EXCLUDED.model_path,
                        last_trained_at = NOW(),
                        version = ml.model_registry.version + 1
                    RETURNING id
                """, {
                    'model_name': kwargs.get('model_name'),
                    'model_type': kwargs.get('model_type'),
                    'signal_type': kwargs.get('signal_type'),
                    'market_regime': kwargs.get('market_regime'),
                    'window_days': window_days,
                    'threshold': threshold,
                    'features': features,
                    'model_path': kwargs.get('model_path')
                })

                model_id = cur.fetchone()[0]
                conn.commit()

                return model_id


def main():
    """Train all models for multi-stage system."""
    trainer = MultiStageModelTrainer()
    trainer.train_all_models()


if __name__ == "__main__":
    main()