"""
ML Prediction Service for Trading Signals
==========================================
Production service for making predictions on new trading signals
using trained models.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions on new trading signals."""

    def __init__(self, model_path: str = 'models/'):
        """
        Initialize prediction service.

        Args:
            model_path: Path to directory containing saved models
        """
        self.model_path = model_path
        self.models = {}
        self.preprocessors = {}

        # Load environment variables
        load_dotenv()

        # Database connection parameters
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Load models and preprocessors
        self._load_models()

    def _load_models(self):
        """Load saved models and preprocessors."""
        try:
            # Load BUY models
            self.models['BUY'] = joblib.load(f'{self.model_path}buy_best_model.pkl')
            self.preprocessors['BUY'] = joblib.load(f'{self.model_path}buy_preprocessor.pkl')
            logger.info("BUY model and preprocessor loaded successfully")

            # Load SELL models
            self.models['SELL'] = joblib.load(f'{self.model_path}sell_best_model.pkl')
            self.preprocessors['SELL'] = joblib.load(f'{self.model_path}sell_preprocessor.pkl')
            logger.info("SELL model and preprocessor loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def get_recent_signals(self, hours_back: int = 1) -> pd.DataFrame:
        """
        Get recent signals from database for prediction.

        Args:
            hours_back: Number of hours to look back for signals

        Returns:
            DataFrame with recent signals
        """
        query = f"""
        SELECT 
            id, timestamp, trading_pair_id, pair_symbol, signal_type,

            -- Core features
            indicator_score, pattern_score, combination_score, total_score,
            signal_strength,

            -- Market features
            close_price, price_change_pct, atr, rsi,

            -- Volume features
            buy_ratio, buy_ratio_weighted, normalized_imbalance, 
            smoothed_imbalance, volume_zscore, cvd_delta, cvd_cumulative,

            -- Future/Funding features
            oi_delta_pct, funding_rate_avg,

            -- Technical indicators
            rs_value, rs_momentum,
            macd_line, macd_signal, macd_histogram,

            -- POC levels
            poc_24h, poc_7d, poc_30d,
            poc_volume_24h, poc_volume_7d,

            -- Pattern and combination details
            patterns_details, combinations_details,

            -- Categorical features
            market_regime, is_meme

        FROM fas.mv_all_signals
        WHERE timestamp >= NOW() - INTERVAL '{hours_back} hours'
            AND timestamp <= NOW()
        ORDER BY timestamp DESC
        """

        try:
            with psycopg2.connect(**self.db_params) as conn:
                df = pd.read_sql(query, conn)
                logger.info(f"Loaded {len(df)} signals from last {hours_back} hours")
                return df
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            raise

    def prepare_signal_features(self, df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
        """
        Prepare features for a specific signal type.

        Args:
            df: DataFrame with raw signals
            signal_type: 'BUY' or 'SELL'

        Returns:
            DataFrame with prepared features
        """
        # Filter by signal type
        df_filtered = df[df['signal_type'] == signal_type].copy()

        if df_filtered.empty:
            return pd.DataFrame()

        # Expand patterns
        for i in range(10):
            df_filtered[f'pattern_{i + 1}_name'] = df_filtered['patterns_details'].apply(
                lambda x: x[i]['pattern'] if x and len(x) > i else None
            )
            df_filtered[f'pattern_{i + 1}_impact'] = df_filtered['patterns_details'].apply(
                lambda x: x[i]['impact'] if x and len(x) > i else None
            )
            df_filtered[f'pattern_{i + 1}_confidence'] = df_filtered['patterns_details'].apply(
                lambda x: x[i]['confidence'] if x and len(x) > i else None
            )

        # Pattern binary features
        df_filtered['has_distribution'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('DISTRIBUTION' in str(p) for p in x) else 0
        )
        df_filtered['has_accumulation'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('ACCUMULATION' in str(p) for p in x) else 0
        )
        df_filtered['has_volume_anomaly'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('VOLUME_ANOMALY' in str(p) for p in x) else 0
        )
        df_filtered['has_momentum_exhaustion'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('MOMENTUM_EXHAUSTION' in str(p) for p in x) else 0
        )
        df_filtered['has_oi_explosion'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('OI_EXPLOSION' in str(p) for p in x) else 0
        )
        df_filtered['has_squeeze_ignition'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('SQUEEZE_IGNITION' in str(p) for p in x) else 0
        )
        df_filtered['has_cvd_divergence'] = df_filtered['patterns_details'].apply(
            lambda x: 1 if x and any('CVD_PRICE_DIVERGENCE' in str(p) for p in x) else 0
        )

        # Expand combinations
        for i in range(6):
            df_filtered[f'combo_{i + 1}_name'] = df_filtered['combinations_details'].apply(
                lambda x: x[i]['combination_name'] if x and len(x) > i else None
            )
            df_filtered[f'combo_{i + 1}_score'] = df_filtered['combinations_details'].apply(
                lambda x: x[i]['score'] if x and len(x) > i else None
            )
            df_filtered[f'combo_{i + 1}_confidence'] = df_filtered['combinations_details'].apply(
                lambda x: x[i]['confidence'] if x and len(x) > i else None
            )

        # Combination binary features
        df_filtered['has_volume_distribution'] = df_filtered['combinations_details'].apply(
            lambda x: 1 if x and any('VOLUME_DISTRIBUTION' in str(c) for c in x) else 0
        )
        df_filtered['has_volume_accumulation'] = df_filtered['combinations_details'].apply(
            lambda x: 1 if x and any('VOLUME_ACCUMULATION' in str(c) for c in x) else 0
        )
        df_filtered['has_institutional_surge'] = df_filtered['combinations_details'].apply(
            lambda x: 1 if x and any('INSTITUTIONAL_SURGE' in str(c) for c in x) else 0
        )
        df_filtered['has_squeeze_momentum'] = df_filtered['combinations_details'].apply(
            lambda x: 1 if x and any('SQUEEZE_MOMENTUM' in str(c) for c in x) else 0
        )
        df_filtered['has_smart_accumulation'] = df_filtered['combinations_details'].apply(
            lambda x: 1 if x and any('SMART_ACCUMULATION' in str(c) for c in x) else 0
        )

        # Signal strength numeric
        strength_map = {'VERY_STRONG': 4, 'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}
        df_filtered['strength_numeric'] = df_filtered['signal_strength'].map(strength_map)

        # Add placeholder for missing target (not needed for prediction)
        df_filtered['target_with_stoploss'] = 0

        return df_filtered

    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for signals.

        Args:
            df: DataFrame with signals

        Returns:
            DataFrame with predictions
        """
        results = []

        for signal_type in ['BUY', 'SELL']:
            # Prepare features
            df_type = self.prepare_signal_features(df, signal_type)

            if df_type.empty:
                continue

            # Get preprocessor and model
            preprocessor = self.preprocessors[signal_type]
            model = self.models[signal_type]

            # Prepare features using preprocessor
            X, _ = preprocessor.prepare_features(df_type, is_training=False)

            # Make predictions
            predictions = model.predict_proba(X)[:, 1]
            predicted_classes = model.predict(X)

            # Add predictions to dataframe
            df_type['win_probability'] = predictions
            df_type['predicted_outcome'] = predicted_classes
            df_type['predicted_outcome_label'] = df_type['predicted_outcome'].map(
                {1: 'TAKE_PROFIT', 0: 'STOP_LOSS'}
            )

            # Calculate expected value
            # Assuming 3% profit on win, -3% loss on loss
            df_type['expected_value'] = df_type['win_probability'] * 3 - (1 - df_type['win_probability']) * 3

            # Add confidence level
            df_type['confidence'] = np.abs(df_type['win_probability'] - 0.5) * 2

            # Add recommendation
            df_type['recommendation'] = df_type.apply(
                lambda row: self._get_recommendation(row), axis=1
            )

            results.append(df_type[['id', 'timestamp', 'pair_symbol', 'signal_type',
                                    'signal_strength', 'total_score', 'win_probability',
                                    'predicted_outcome_label', 'expected_value',
                                    'confidence', 'recommendation']])

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def _get_recommendation(self, row: pd.Series) -> str:
        """
        Get trading recommendation based on prediction.

        Args:
            row: Series with signal data and prediction

        Returns:
            Recommendation string
        """
        if row['expected_value'] > 1.5 and row['confidence'] > 0.7:
            return 'STRONG_TRADE'
        elif row['expected_value'] > 0.5 and row['confidence'] > 0.5:
            return 'TRADE'
        elif row['expected_value'] > 0 and row['confidence'] > 0.3:
            return 'WEAK_TRADE'
        else:
            return 'SKIP'

    def save_predictions(self, predictions: pd.DataFrame,
                         table_name: str = 'ml_predictions'):
        """
        Save predictions to database.

        Args:
            predictions: DataFrame with predictions
            table_name: Name of the table to save predictions
        """
        # Create table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS fas.{table_name} (
            id SERIAL PRIMARY KEY,
            signal_id INTEGER,
            timestamp TIMESTAMPTZ,
            pair_symbol VARCHAR(20),
            signal_type VARCHAR(10),
            signal_strength VARCHAR(20),
            total_score NUMERIC,
            win_probability NUMERIC(5,4),
            predicted_outcome VARCHAR(20),
            expected_value NUMERIC(8,4),
            confidence NUMERIC(5,4),
            recommendation VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
            ON fas.{table_name}(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_{table_name}_recommendation 
            ON fas.{table_name}(recommendation);
        """

        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Create table
                    cur.execute(create_table_query)

                    # Insert predictions
                    for _, row in predictions.iterrows():
                        insert_query = f"""
                        INSERT INTO fas.{table_name} 
                        (signal_id, timestamp, pair_symbol, signal_type, signal_strength,
                         total_score, win_probability, predicted_outcome, expected_value,
                         confidence, recommendation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cur.execute(insert_query, (
                            row['id'], row['timestamp'], row['pair_symbol'],
                            row['signal_type'], row['signal_strength'], row['total_score'],
                            row['win_probability'], row['predicted_outcome_label'],
                            row['expected_value'], row['confidence'], row['recommendation']
                        ))

                    conn.commit()
                    logger.info(f"Saved {len(predictions)} predictions to {table_name}")

        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise

    def get_top_opportunities(self, n: int = 10) -> pd.DataFrame:
        """
        Get top trading opportunities from recent predictions.

        Args:
            n: Number of top opportunities to return

        Returns:
            DataFrame with top opportunities
        """
        # Get recent signals
        df = self.get_recent_signals(hours_back=1)

        if df.empty:
            logger.warning("No recent signals found")
            return pd.DataFrame()

        # Make predictions
        predictions = self.predict_signals(df)

        if predictions.empty:
            logger.warning("No predictions generated")
            return pd.DataFrame()

        # Filter and sort by expected value
        opportunities = predictions[
            predictions['recommendation'].isin(['STRONG_TRADE', 'TRADE'])
        ].sort_values('expected_value', ascending=False).head(n)

        return opportunities

    def run_continuous_prediction(self, interval_minutes: int = 15):
        """
        Run continuous prediction service.

        Args:
            interval_minutes: Interval between prediction runs in minutes
        """
        import time

        logger.info(f"Starting continuous prediction service (interval: {interval_minutes} minutes)")

        while True:
            try:
                # Get and predict recent signals
                df = self.get_recent_signals(hours_back=1)

                if not df.empty:
                    predictions = self.predict_signals(df)

                    if not predictions.empty:
                        # Save predictions
                        self.save_predictions(predictions)

                        # Log top opportunities
                        top_opps = predictions[
                            predictions['recommendation'].isin(['STRONG_TRADE', 'TRADE'])
                        ].head(5)

                        if not top_opps.empty:
                            logger.info("\nTop Trading Opportunities:")
                            for _, row in top_opps.iterrows():
                                logger.info(
                                    f"  {row['pair_symbol']} - {row['signal_type']} - "
                                    f"Win Prob: {row['win_probability']:.2%} - "
                                    f"Expected: {row['expected_value']:.2f}% - "
                                    f"{row['recommendation']}"
                                )
                        else:
                            logger.info("No strong trading opportunities at this time")
                else:
                    logger.info("No new signals to process")

                # Wait for next iteration
                logger.info(f"Waiting {interval_minutes} minutes until next run...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Stopping prediction service...")
                break
            except Exception as e:
                logger.error(f"Error in prediction cycle: {e}")
                logger.info(f"Retrying in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)


class BacktestService:
    """Service for backtesting predictions."""

    def __init__(self, prediction_service: PredictionService):
        """
        Initialize backtest service.

        Args:
            prediction_service: Instance of PredictionService
        """
        self.prediction_service = prediction_service
        self.db_params = prediction_service.db_params

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with backtest results
        """
        query = f"""
        SELECT *
        FROM fas.ml_new_training_data
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
            AND target_with_stoploss IS NOT NULL
        ORDER BY timestamp
        """

        try:
            with psycopg2.connect(**self.db_params) as conn:
                df = pd.read_sql(query, conn)

            logger.info(f"Loaded {len(df)} signals for backtesting")

            # Make predictions
            predictions = self.prediction_service.predict_signals(df)

            # Merge with actual outcomes
            results = pd.merge(
                predictions,
                df[['id', 'target_with_stoploss']],
                on='id',
                how='inner'
            )

            # Calculate metrics
            metrics = self._calculate_backtest_metrics(results)

            return {
                'metrics': metrics,
                'results': results,
                'summary': self._generate_summary(metrics)
            }

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    def _calculate_backtest_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate backtest metrics.

        Args:
            results: DataFrame with predictions and actual outcomes

        Returns:
            Dictionary with metrics
        """
        # Filter trades based on recommendation
        trades = results[results['recommendation'].isin(['STRONG_TRADE', 'TRADE'])]

        if trades.empty:
            return {'error': 'No trades generated'}

        # Calculate metrics
        metrics = {
            'total_signals': len(results),
            'total_trades': len(trades),
            'trade_rate': len(trades) / len(results),
            'actual_win_rate': trades['target_with_stoploss'].mean(),
            'predicted_win_rate': trades['win_probability'].mean(),
            'accuracy': (trades['predicted_outcome'] == trades['target_with_stoploss']).mean(),
            'avg_expected_value': trades['expected_value'].mean(),
            'actual_profit': trades['target_with_stoploss'].sum() * 3 - (1 - trades['target_with_stoploss']).sum() * 3,
            'expected_profit': trades['expected_value'].sum()
        }

        # Calculate by signal type
        for signal_type in ['BUY', 'SELL']:
            type_trades = trades[trades['signal_type'] == signal_type]
            if not type_trades.empty:
                metrics[f'{signal_type}_trades'] = len(type_trades)
                metrics[f'{signal_type}_win_rate'] = type_trades['target_with_stoploss'].mean()
                metrics[f'{signal_type}_accuracy'] = (
                        type_trades['predicted_outcome'] == type_trades['target_with_stoploss']
                ).mean()

        return metrics

    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate backtest summary.

        Args:
            metrics: Dictionary with metrics

        Returns:
            Summary string
        """
        if 'error' in metrics:
            return metrics['error']

        summary = f"""
        Backtest Summary
        ================
        Total Signals: {metrics['total_signals']}
        Total Trades: {metrics['total_trades']} ({metrics['trade_rate']:.1%})

        Performance:
        - Actual Win Rate: {metrics['actual_win_rate']:.1%}
        - Predicted Win Rate: {metrics['predicted_win_rate']:.1%}
        - Accuracy: {metrics['accuracy']:.1%}

        Profitability:
        - Expected Profit: {metrics['expected_profit']:.2f}%
        - Actual Profit: {metrics['actual_profit']:.2f}%
        - Avg Expected Value: {metrics['avg_expected_value']:.2f}%

        BUY Signals:
        - Trades: {metrics.get('BUY_trades', 0)}
        - Win Rate: {metrics.get('BUY_win_rate', 0):.1%}
        - Accuracy: {metrics.get('BUY_accuracy', 0):.1%}

        SELL Signals:
        - Trades: {metrics.get('SELL_trades', 0)}
        - Win Rate: {metrics.get('SELL_win_rate', 0):.1%}
        - Accuracy: {metrics.get('SELL_accuracy', 0):.1%}
        """

        return summary


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='ML Trading Prediction Service')
    parser.add_argument('--mode', choices=['predict', 'continuous', 'backtest', 'top'],
                        default='predict', help='Service mode')
    parser.add_argument('--hours', type=int, default=1,
                        help='Hours to look back for signals')
    parser.add_argument('--interval', type=int, default=15,
                        help='Interval for continuous mode (minutes)')
    parser.add_argument('--start-date', type=str,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top opportunities to show')

    args = parser.parse_args()

    # Initialize service
    service = PredictionService()

    if args.mode == 'predict':
        # Single prediction run
        df = service.get_recent_signals(hours_back=args.hours)
        if not df.empty:
            predictions = service.predict_signals(df)
            print("\nPredictions:")
            print(predictions.to_string())
            service.save_predictions(predictions)
        else:
            print("No signals found")

    elif args.mode == 'continuous':
        # Continuous prediction service
        service.run_continuous_prediction(interval_minutes=args.interval)

    elif args.mode == 'backtest':
        # Backtest mode
        if not args.start_date or not args.end_date:
            print("Please provide --start-date and --end-date for backtest")
            return

        backtest = BacktestService(service)
        results = backtest.run_backtest(args.start_date, args.end_date)
        print(results['summary'])

    elif args.mode == 'top':
        # Show top opportunities
        opportunities = service.get_top_opportunities(n=args.top_n)
        if not opportunities.empty:
            print(f"\nTop {args.top_n} Trading Opportunities:")
            print(opportunities.to_string())
        else:
            print("No trading opportunities found")


if __name__ == "__main__":
    main()