#!/usr/bin/env python3
"""
Quick fix for remaining issues
"""

import psycopg2
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:

            # 1. Fix SQL function
            logger.info("Fixing check_system_health function...")
            cur.execute("""
                CREATE OR REPLACE FUNCTION ml.check_system_health()
                RETURNS TABLE (
                    check_name VARCHAR,
                    status VARCHAR,
                    details TEXT
                ) AS $$
                BEGIN
                    -- Check 1: Models availability
                    RETURN QUERY
                    SELECT 
                        'Models Availability'::VARCHAR,
                        CASE 
                            WHEN COUNT(*) >= 6 THEN 'OK'::VARCHAR
                            ELSE 'WARNING'::VARCHAR
                        END,
                        'Active models: ' || COUNT(*)::TEXT || ' (need at least 6)'
                    FROM ml.model_registry
                    WHERE is_active = true;

                    -- Check 2: Recent predictions
                    RETURN QUERY
                    SELECT 
                        'Recent Predictions'::VARCHAR,
                        CASE 
                            WHEN COUNT(*) > 0 THEN 'OK'::VARCHAR
                            ELSE 'ERROR'::VARCHAR
                        END,
                        'Predictions in last hour: ' || COUNT(*)::TEXT
                    FROM ml.multistage_predictions
                    WHERE predicted_at >= NOW() - INTERVAL '1 hour';

                    -- Check 3: Win rate
                    RETURN QUERY
                    SELECT 
                        'Win Rate'::VARCHAR,
                        CASE 
                            WHEN COUNT(*) = 0 THEN 'NO DATA'::VARCHAR
                            WHEN AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) >= 0.70 THEN 'OK'::VARCHAR
                            WHEN AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) >= 0.60 THEN 'WARNING'::VARCHAR
                            ELSE 'ERROR'::VARCHAR
                        END,
                        CASE 
                            WHEN COUNT(*) = 0 THEN 'No outcome data yet'
                            ELSE '24h win rate: ' || 
                                ROUND(AVG(CASE WHEN actual_outcome THEN 1 ELSE 0 END) * 100, 1)::TEXT || '%'
                        END
                    FROM ml.multistage_predictions
                    WHERE final_decision = true
                        AND actual_outcome IS NOT NULL
                        AND predicted_at >= NOW() - INTERVAL '24 hours';

                    -- Check 4: Pass rate
                    RETURN QUERY
                    SELECT 
                        'Signal Pass Rate'::VARCHAR,
                        CASE 
                            WHEN COUNT(*) = 0 THEN 'NO DATA'::VARCHAR
                            WHEN AVG(CASE WHEN final_decision THEN 1 ELSE 0 END) BETWEEN 0.05 AND 0.15 THEN 'OK'::VARCHAR
                            WHEN AVG(CASE WHEN final_decision THEN 1 ELSE 0 END) BETWEEN 0.01 AND 0.20 THEN 'WARNING'::VARCHAR
                            ELSE 'ERROR'::VARCHAR
                        END,
                        CASE
                            WHEN COUNT(*) = 0 THEN 'No predictions yet'
                            ELSE 'Final pass rate: ' || 
                                ROUND(AVG(CASE WHEN final_decision THEN 1 ELSE 0 END) * 100, 1)::TEXT || 
                                '% (target: 5-15%)'
                        END
                    FROM ml.multistage_predictions
                    WHERE predicted_at >= NOW() - INTERVAL '24 hours';

                END;
                $$ LANGUAGE plpgsql;
            """)
            conn.commit()
            logger.info("✅ Fixed check_system_health function")

            # 2. Set VERY low thresholds for Stage 4
            logger.info("\nSetting ultra-low thresholds for Stage 4...")
            cur.execute("""
                UPDATE ml.model_registry
                SET threshold = CASE 
                    WHEN model_type = 'quick_filter' THEN 0.20
                    WHEN model_type = 'regime_specific' THEN 0.03  -- Ultra low
                    WHEN model_type = 'precision' THEN 0.05        -- Ultra low for Stage 4
                    ELSE threshold
                END
                WHERE is_active = true
                RETURNING model_name, model_type, threshold
            """)

            updates = cur.fetchall()
            for name, type_, threshold in updates:
                logger.info(f"  {name} ({type_}): {threshold:.3f}")

            conn.commit()
            logger.info(f"✅ Updated {len(updates)} model thresholds")

            # 3. Test the health check
            logger.info("\nTesting health check...")
            cur.execute("SELECT * FROM ml.check_system_health()")
            for row in cur.fetchall():
                check, status, details = row
                emoji = "✅" if status == "OK" else "⚠️" if status in ["WARNING", "NO DATA"] else "❌"
                logger.info(f"  {emoji} {check}: {details}")

            logger.info("\n✅ All fixes applied!")
            logger.info("\nNow run: python final_test.py")


if __name__ == "__main__":
    main()