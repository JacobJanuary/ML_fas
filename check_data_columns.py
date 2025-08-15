"""
Quick script to check what columns are in the training data
and identify non-numeric ones.
"""

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn_params = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Load sample data
query = """
SELECT *
FROM fas.ml_training_data_direct
WHERE signal_type = 'BUY'
LIMIT 100
"""

with psycopg2.connect(**conn_params) as conn:
    df = pd.read_sql(query, conn)

print("=" * 60)
print("DATA COLUMNS ANALYSIS")
print("=" * 60)

print(f"\nTotal columns: {len(df.columns)}")
print("\nColumn types:")
print(df.dtypes.value_counts())

print("\n" + "=" * 60)
print("NUMERIC COLUMNS:")
print("=" * 60)
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
for col in numeric_cols:
    print(f"  - {col}")

print("\n" + "=" * 60)
print("NON-NUMERIC COLUMNS:")
print("=" * 60)
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
for col in non_numeric_cols:
    print(f"  - {col}: {df[col].dtype}")
    # Show unique values if not too many
    unique_vals = df[col].nunique()
    if unique_vals < 10:
        print(f"    Unique values: {df[col].unique()[:5]}")

print("\n" + "=" * 60)
print("COLUMNS WITH TEXT VALUES:")
print("=" * 60)
for col in df.columns:
    if df[col].dtype == 'object':
        sample_values = df[col].dropna().unique()[:3]
        print(f"  {col}: {sample_values}")