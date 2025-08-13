"""
Analyze temporal distribution shift in ML training data
"""

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()


def analyze_temporal_distribution():
    """
    Analyze how win rate changes over time
    """

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    # Load data
    query = """
    SELECT 
        timestamp,
        signal_type,
        market_regime,
        target,
        total_score,
        signal_strength,
        price_to_poc_24h_pct,
        volume_zscore,
        rsi
    FROM fas.ml_training_data_v2
    ORDER BY timestamp
    """

    print("Loading data...")
    df = pd.read_sql(query, conn)
    conn.close()

    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    # Calculate daily statistics
    daily_stats = df.groupby(['date', 'signal_type']).agg({
        'target': ['mean', 'count'],
        'total_score': 'mean',
        'volume_zscore': 'mean',
        'rsi': 'mean'
    }).reset_index()

    daily_stats.columns = ['date', 'signal_type', 'win_rate', 'count', 'avg_score', 'avg_volume_z', 'avg_rsi']

    # Calculate regime distribution by date
    regime_dist = df.groupby(['date', 'market_regime']).size().unstack(fill_value=0)
    regime_pct = regime_dist.div(regime_dist.sum(axis=1), axis=0) * 100

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 1. Win rate over time
    ax = axes[0]
    for signal_type in ['BUY', 'SELL']:
        data = daily_stats[daily_stats['signal_type'] == signal_type]
        ax.plot(data['date'], data['win_rate'] * 100, label=f'{signal_type} Win Rate', marker='o', markersize=3)

    # Add temporal split lines
    total_days = len(daily_stats['date'].unique())
    train_end = daily_stats['date'].unique()[int(total_days * 0.7)]
    val_end = daily_stats['date'].unique()[int(total_days * 0.85)]

    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.5, label='Train/Val Split')
    ax.axvline(x=val_end, color='orange', linestyle='--', alpha=0.5, label='Val/Test Split')

    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate Changes Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Signal count over time
    ax = axes[1]
    for signal_type in ['BUY', 'SELL']:
        data = daily_stats[daily_stats['signal_type'] == signal_type]
        ax.plot(data['date'], data['count'], label=f'{signal_type} Count', marker='o', markersize=3)

    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=val_end, color='orange', linestyle='--', alpha=0.5)

    ax.set_ylabel('Signal Count')
    ax.set_title('Number of Signals Per Day')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Market regime distribution
    ax = axes[2]
    regime_pct.plot(kind='area', stacked=True, ax=ax, alpha=0.7)

    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=val_end, color='orange', linestyle='--', alpha=0.5)

    ax.set_ylabel('Regime Distribution (%)')
    ax.set_title('Market Regime Distribution Over Time')
    ax.legend(title='Market Regime')
    ax.grid(True, alpha=0.3)

    # 4. Average indicators
    ax = axes[3]
    buy_data = daily_stats[daily_stats['signal_type'] == 'BUY']
    ax.plot(buy_data['date'], buy_data['avg_rsi'], label='Avg RSI (BUY)', alpha=0.7)
    ax.plot(buy_data['date'], buy_data['avg_volume_z'] * 10 + 50, label='Avg Volume Z-score x10 + 50 (BUY)', alpha=0.7)

    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=val_end, color='orange', linestyle='--', alpha=0.5)

    ax.set_ylabel('Value')
    ax.set_title('Average Technical Indicators Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('temporal_distribution_analysis.png')
    plt.show()

    # Print statistics for each period
    print("\n" + "=" * 60)
    print("TEMPORAL DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Define periods
    train_mask = df['date'] <= train_end
    val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
    test_mask = df['date'] > val_end

    for signal_type in ['BUY', 'SELL']:
        print(f"\n{signal_type} Signal Analysis:")
        print("-" * 40)

        signal_mask = df['signal_type'] == signal_type

        # Train period
        train_data = df[signal_mask & train_mask]
        print(f"Train Period ({train_data['date'].min()} to {train_data['date'].max()}):")
        print(f"  Samples: {len(train_data):,}")
        print(f"  Win Rate: {train_data['target'].mean() * 100:.1f}%")
        print(f"  Market Regime:")
        for regime, count in train_data['market_regime'].value_counts().items():
            pct = count / len(train_data) * 100
            print(f"    {regime}: {pct:.1f}%")

        # Validation period
        val_data = df[signal_mask & val_mask]
        print(f"\nValidation Period ({val_data['date'].min()} to {val_data['date'].max()}):")
        print(f"  Samples: {len(val_data):,}")
        print(f"  Win Rate: {val_data['target'].mean() * 100:.1f}%")
        print(f"  Market Regime:")
        for regime, count in val_data['market_regime'].value_counts().items():
            pct = count / len(val_data) * 100
            print(f"    {regime}: {pct:.1f}%")

        # Test period
        test_data = df[signal_mask & test_mask]
        print(f"\nTest Period ({test_data['date'].min()} to {test_data['date'].max()}):")
        print(f"  Samples: {len(test_data):,}")
        print(f"  Win Rate: {test_data['target'].mean() * 100:.1f}%")
        print(f"  Market Regime:")
        for regime, count in test_data['market_regime'].value_counts().items():
            pct = count / len(test_data) * 100
            print(f"    {regime}: {pct:.1f}%")

    # Analyze extreme days
    print("\n" + "=" * 60)
    print("EXTREME WIN RATE DAYS")
    print("=" * 60)

    for signal_type in ['BUY', 'SELL']:
        signal_stats = daily_stats[daily_stats['signal_type'] == signal_type]

        # Find extremes
        best_days = signal_stats.nlargest(3, 'win_rate')
        worst_days = signal_stats.nsmallest(3, 'win_rate')

        print(f"\n{signal_type} Best Days:")
        for _, row in best_days.iterrows():
            print(f"  {row['date']}: {row['win_rate'] * 100:.1f}% win rate ({row['count']} signals)")

        print(f"\n{signal_type} Worst Days:")
        for _, row in worst_days.iterrows():
            print(f"  {row['date']}: {row['win_rate'] * 100:.1f}% win rate ({row['count']} signals)")

    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION WITH WIN RATE")
    print("=" * 60)

    # Calculate correlations for each signal type
    for signal_type in ['BUY', 'SELL']:
        signal_data = df[df['signal_type'] == signal_type].copy()

        # Convert market regime to numeric
        regime_map = {'BULL': 1, 'NEUTRAL': 0, 'BEAR': -1}
        signal_data['regime_numeric'] = signal_data['market_regime'].map(regime_map)

        # Calculate correlations
        correlations = signal_data[['target', 'total_score', 'volume_zscore', 'rsi', 'regime_numeric']].corr()[
            'target'].drop('target')

        print(f"\n{signal_type} Correlations with Win Rate:")
        for feature, corr in correlations.sort_values(ascending=False).items():
            print(f"  {feature}: {corr:.3f}")

    print("\n" + "=" * 60)
    print("✅ Analysis Complete - Check temporal_distribution_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    analyze_temporal_distribution()