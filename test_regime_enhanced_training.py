"""
Check if market regime function exists and how to use it
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

def check_regime_function():
    """Check market regime function availability."""

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("="*60)
    print("CHECKING MARKET REGIME FUNCTION")
    print("="*60)

    # 1. Check if function exists
    cur.execute("""
        SELECT 
            routine_name,
            routine_type,
            data_type
        FROM information_schema.routines
        WHERE routine_schema = 'fas'
            AND routine_name LIKE '%market_regime%'
    """)

    functions = cur.fetchall()
    if functions:
        print("\n✅ Found market regime functions:")
        for func in functions:
            print(f"  - {func['routine_name']} ({func['routine_type']})")
    else:
        print("\n❌ No market regime functions found")

    # 2. Check function parameters
    cur.execute("""
        SELECT 
            p.proname as function_name,
            pg_get_function_arguments(p.oid) as arguments,
            pg_get_function_result(p.oid) as return_type
        FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'fas'
            AND p.proname LIKE '%market_regime%'
    """)

    func_details = cur.fetchall()
    if func_details:
        print("\n📝 Function signatures:")
        for func in func_details:
            print(f"  {func['function_name']}({func['arguments']}) -> {func['return_type']}")

    # 3. Try different ways to call the function
    print("\n🔍 Testing function calls:")

    # Test 1: Try with NOW()
    try:
        cur.execute("""
            SELECT fas.get_market_regime_at_time(NOW(), '4h'::fas.timeframe_enum) as regime
        """)
        result = cur.fetchone()
        if result and result['regime']:
            print(f"  ✅ Current regime (4h): {result['regime']}")
        else:
            print(f"  ⚠️ Function returned NULL for current time")
    except Exception as e:
        print(f"  ❌ Error with NOW(): {str(e)[:100]}")

    # Test 2: Try with specific timestamp
    try:
        cur.execute("""
            SELECT fas.get_market_regime_at_time(
                '2025-08-10 12:00:00+00'::timestamp with time zone,
                '4h'::fas.timeframe_enum
            ) as regime
        """)
        result = cur.fetchone()
        if result and result['regime']:
            print(f"  ✅ Regime for 2025-08-10 12:00: {result['regime']}")
        else:
            print(f"  ⚠️ Function returned NULL for specific timestamp")
    except Exception as e:
        print(f"  ❌ Error with specific timestamp: {str(e)[:100]}")

    # Test 3: Try without timeframe parameter
    try:
        cur.execute("""
            SELECT fas.get_market_regime_at_time(NOW()) as regime
        """)
        result = cur.fetchone()
        if result and result['regime']:
            print(f"  ✅ Current regime (default): {result['regime']}")
        else:
            print(f"  ⚠️ Function returned NULL without timeframe")
    except Exception as e:
        print(f"  ❌ Error without timeframe: {str(e)[:100]}")

    # 4. Check if market_regime_history table exists
    cur.execute("""
        SELECT COUNT(*) as count
        FROM information_schema.tables
        WHERE table_schema = 'fas'
            AND table_name LIKE '%market_regime%'
    """)

    result = cur.fetchone()
    if result and result['count'] > 0:
        print(f"\n✅ Found market regime related tables")

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'fas'
                AND table_name LIKE '%market_regime%'
        """)

        for row in cur.fetchall():
            print(f"  - {row['table_name']}")

    # 5. Alternative: Get regime from existing data
    print("\n📊 Alternative: Get regime from ml_training_data_v2")
    cur.execute("""
        SELECT 
            market_regime,
            COUNT(*) as count,
            MAX(timestamp) as latest
        FROM fas.ml_training_data_v2
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY market_regime
    """)

    recent_regimes = cur.fetchall()
    if recent_regimes:
        print("  Recent market regimes:")
        for regime in recent_regimes:
            print(f"    {regime['market_regime']}: {regime['count']} signals (latest: {regime['latest']})")
    else:
        print("  No recent data in last hour")

    conn.close()

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("If function doesn't exist or returns NULL:")
    print("1. Use market_regime from ml_training_data_v2 directly")
    print("2. Or implement simple regime detection based on price movement")
    print("3. Or check if function needs different parameters")


if __name__ == "__main__":
    check_regime_function()