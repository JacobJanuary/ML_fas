"""
Helper functions for working with market regime
"""

import psycopg2
from typing import Dict, Tuple, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class MarketRegimeHelper:
    """Helper class for market regime operations."""

    def __init__(self):
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

    def get_market_regime(self, timestamp: datetime, timeframe: str = '4h') -> Dict[str, any]:
        """
        Get market regime with all details.

        Args:
            timestamp: Timestamp to check
            timeframe: Timeframe (15m, 1h, 4h)

        Returns:
            Dictionary with regime details
        """
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM fas.get_market_regime_at_time(%s, %s::fas.timeframe_enum)
                """, (timestamp, timeframe))

                result = cur.fetchone()

                if result:
                    # Function returns: (regime, strength, btc_trend, alt_trend, volume_trend, data_timestamp)
                    return {
                        'regime': result[0],  # BULL, BEAR, or NEUTRAL
                        'strength': float(result[1]) if result[1] else 0,
                        'btc_trend': float(result[2]) if result[2] else 0,
                        'alt_trend': float(result[3]) if result[3] else 0,
                        'volume_trend': float(result[4]) if result[4] else 0,
                        'data_timestamp': result[5]
                    }
                else:
                    return {
                        'regime': 'NEUTRAL',
                        'strength': 0,
                        'btc_trend': 0,
                        'alt_trend': 0,
                        'volume_trend': 0,
                        'data_timestamp': None
                    }

    def get_regime_simple(self, timestamp: datetime, timeframe: str = '4h') -> str:
        """
        Get just the regime name (simplified).

        Args:
            timestamp: Timestamp to check
            timeframe: Timeframe

        Returns:
            Regime name: BULL, BEAR, or NEUTRAL
        """
        regime_data = self.get_market_regime(timestamp, timeframe)
        return regime_data['regime']

    def should_take_trade(self, signal_type: str, timestamp: datetime,
                          ml_probability: float, timeframe: str = '4h') -> Tuple[bool, str]:
        """
        Determine if trade should be taken based on regime.

        Args:
            signal_type: BUY or SELL
            timestamp: Signal timestamp
            ml_probability: ML model probability
            timeframe: Regime timeframe

        Returns:
            Tuple of (should_trade, reason)
        """
        regime_data = self.get_market_regime(timestamp, timeframe)
        regime = regime_data['regime']
        strength = regime_data['strength']

        # Strong regime alignment
        if signal_type == 'BUY':
            if regime == 'BULL':
                if strength > 2 and ml_probability > 0.6:
                    return True, f"STRONG BUY: Bull regime (strength {strength:.2f})"
                elif ml_probability > 0.65:
                    return True, f"BUY: Bull regime aligned"
                else:
                    return False, f"Probability too low ({ml_probability:.2%})"

            elif regime == 'NEUTRAL':
                if ml_probability > 0.7:
                    return True, f"CONSIDER: High probability in neutral"
                else:
                    return False, f"Neutral regime, need higher confidence"

            else:  # BEAR
                return False, f"SKIP: Buy signal in Bear regime"

        else:  # SELL
            if regime == 'BEAR':
                if strength > 2 and ml_probability > 0.6:
                    return True, f"STRONG SELL: Bear regime (strength {strength:.2f})"
                elif ml_probability > 0.65:
                    return True, f"SELL: Bear regime aligned"
                else:
                    return False, f"Probability too low ({ml_probability:.2%})"

            elif regime == 'NEUTRAL':
                if ml_probability > 0.7:
                    return True, f"CONSIDER: High probability in neutral"
                else:
                    return False, f"Neutral regime, need higher confidence"

            else:  # BULL
                return False, f"SKIP: Sell signal in Bull regime"

    def analyze_regime_quality(self, timestamp: datetime = None) -> Dict:
        """
        Analyze current or historical regime quality.

        Args:
            timestamp: Optional timestamp (default: now)

        Returns:
            Analysis dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()

        results = {}

        for timeframe in ['15m', '1h', '4h']:
            regime_data = self.get_market_regime(timestamp, timeframe)

            results[timeframe] = {
                'regime': regime_data['regime'],
                'strength': regime_data['strength'],
                'quality': self._assess_quality(regime_data),
                'confidence': self._calculate_confidence(regime_data)
            }

        # Determine consensus
        regimes = [r['regime'] for r in results.values()]
        if all(r == regimes[0] for r in regimes):
            results['consensus'] = f"STRONG {regimes[0]}"
        elif regimes.count(regimes[0]) >= 2:
            results['consensus'] = f"MODERATE {max(set(regimes), key=regimes.count)}"
        else:
            results['consensus'] = "MIXED"

        return results

    def _assess_quality(self, regime_data: Dict) -> str:
        """Assess regime quality based on strength."""
        strength = abs(regime_data['strength'])

        if strength > 3:
            return "VERY_STRONG"
        elif strength > 2:
            return "STRONG"
        elif strength > 1:
            return "MODERATE"
        elif strength > 0.5:
            return "WEAK"
        else:
            return "VERY_WEAK"

    def _calculate_confidence(self, regime_data: Dict) -> float:
        """Calculate confidence in regime (0-1)."""
        strength = abs(regime_data['strength'])

        # Normalize strength to 0-1 range
        # Assuming strength rarely exceeds 5
        confidence = min(strength / 5, 1.0)

        # Factor in component alignment
        components = [
            abs(regime_data['btc_trend']),
            abs(regime_data['alt_trend']),
            abs(regime_data['volume_trend'])
        ]

        # Check if all components align (same sign)
        if regime_data['btc_trend'] * regime_data['alt_trend'] > 0:
            confidence *= 1.1  # Boost for alignment

        return min(confidence, 1.0)


def demo_usage():
    """Demonstrate usage of MarketRegimeHelper."""

    helper = MarketRegimeHelper()

    print("=" * 60)
    print("MARKET REGIME HELPER DEMO")
    print("=" * 60)

    # Test current regime
    current_regime = helper.get_market_regime(datetime.now(), '4h')
    print(f"\nCurrent 4h Regime:")
    print(f"  Regime: {current_regime['regime']}")
    print(f"  Strength: {current_regime['strength']:.2f}")
    print(f"  BTC Trend: {current_regime['btc_trend']:.2f}")
    print(f"  Alt Trend: {current_regime['alt_trend']:.2f}")
    print(f"  Volume Trend: {current_regime['volume_trend']:.2f}")

    # Test trade decision
    print(f"\n📊 Trade Decision Examples:")

    test_cases = [
        ('BUY', 0.65),
        ('BUY', 0.55),
        ('SELL', 0.70),
        ('SELL', 0.60)
    ]

    for signal_type, probability in test_cases:
        should_trade, reason = helper.should_take_trade(
            signal_type, datetime.now(), probability
        )

        print(f"\n  {signal_type} with {probability:.0%} probability:")
        print(f"    Decision: {'✅ TRADE' if should_trade else '❌ SKIP'}")
        print(f"    Reason: {reason}")

    # Analyze regime quality
    print(f"\n🔍 Regime Quality Analysis:")
    analysis = helper.analyze_regime_quality()

    for timeframe, data in analysis.items():
        if timeframe != 'consensus':
            print(f"\n  {timeframe}:")
            print(f"    Regime: {data['regime']}")
            print(f"    Quality: {data['quality']}")
            print(f"    Confidence: {data['confidence']:.1%}")

    print(f"\n  Overall Consensus: {analysis['consensus']}")


if __name__ == "__main__":
    demo_usage()