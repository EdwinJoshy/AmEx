import numpy as np
import pandas as pd

# Step 1: Load Data
df = pd.read_parquet('amex_offers_data.parquet')

# Step 2: Define data types
pandas_dtypes = {
    'customer_id': 'object',
    'offer_id': 'category',
    'event_ts': 'datetime64[ns]',
    'event_dt': 'datetime64[ns]',
    'offer_action': 'object',
}
for i in range(1, 44):
    pandas_dtypes[f'var_{i}'] = 'float64'
for i in range(44, 51):
    pandas_dtypes[f'var_{i}'] = 'int64'

# Step 3: Apply data type conversions safely
for col, dtype in pandas_dtypes.items():
    if col in df.columns:
        try:
            if dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif 'float' in dtype or 'int' in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'category':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"Couldn't convert {col} to {dtype}: {e}")

# Step 4: Handle missing values properly

# Replace blank strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Numeric columns → fill with 0
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(0)

# Categorical/Object columns → fill with 'Unknown'
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].dtype.name == 'category' and 'Unknown' not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(['Unknown'])
    df[col] = df[col].fillna('Unknown')

# Datetime columns → fill with fixed date
for col in df.select_dtypes(include=['datetime64[ns]']).columns:
    df[col] = df[col].fillna(pd.Timestamp('2000-01-01'))

# Step 5: Clean specific column
# Convert TRUE/FALSE (as string or bool) to 1/0
# Step 1: Normalize to lowercase strings
df['offer_action'] = df['offer_action'].astype(str).str.strip().str.lower()

# Step 2: Replace all valid true/false values
df['offer_action'] = df['offer_action'].replace({
    'true': 1, '1': 1, 'yes': 1,
    'false': 0, '0': 0, 'no': 0,
    'nan': np.nan  # preserve missing if any
})

# Step 3: Convert to int (optional, if no NaNs remain)
df['offer_action'] = pd.to_numeric(df['offer_action'], errors='coerce').fillna(0).astype(int)


# Step 6: Final cleanup
df = df.drop_duplicates()
df = df.reset_index(drop=True)


# Step 7: Confirm
print("✅ Data cleaned successfully!")
print(df.info())
print(df.dtypes)
df.to_csv('temp.csv',index=False)
