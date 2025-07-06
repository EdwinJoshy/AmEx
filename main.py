import numpy as np
import pandas as pd
from dateutil import parser

# Step 1: Load Data
df = pd.read_parquet('amex_offers_data.parquet')

# Step 2: Define static dtypes
pandas_dtypes = {
    'customer_id': 'object',
    'offer_id': 'category',
    'event_ts': 'object',   # parse separately
    'event_dt': 'object',   # parse separately
    'offer_action': 'object',
}

# Dynamically assign remaining var types
for i in range(1, 44):
    pandas_dtypes[f'var_{i}'] = 'float64'
for i in range(44, 51):
    col = f'var_{i}'
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({'true': 1, 'false': 0, 'nan': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    pandas_dtypes[col] = 'int64'

# Step 3: Apply dtypes
for col, dtype in pandas_dtypes.items():
    if col in df.columns:
        try:
            if dtype == 'category':
                df[col] = df[col].astype('category')
            elif dtype in ['float64', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            print(f"Couldn't convert {col} to {dtype}: {e}")

# Step 4: Replace blanks with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Step 5: Fill numeric missing values
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(0)

# Step 6: Fill categorical missing with 'Unknown'
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].dtype.name == 'category' and 'Unknown' not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(['Unknown'])
    df[col] = df[col].fillna('Unknown')

# Step 7: Clean offer_action to binary
df['offer_action'] = df['offer_action'].astype(str).str.strip().str.lower()
df['offer_action'] = df['offer_action'].replace({
    'true': 1, '1': 1, 'yes': 1,
    'false': 0, '0': 0, 'no': 0,
    'nan': np.nan
})
df['offer_action'] = pd.to_numeric(df['offer_action'], errors='coerce').fillna(0).astype(int)

# Step 8: Clean event_dt to DD-MM-YYYY
def clean_event_date(val):
    try:
        return parser.parse(val, dayfirst=False).strftime('%d-%m-%Y')
    except:
        return np.nan

if 'event_dt' in df.columns:
    df['event_dt'] = df['event_dt'].astype(str).str.strip()
    df['event_dt'] = df['event_dt'].apply(clean_event_date)

# Step 9: Clean event_ts to HH:MM:SS
def clean_event_time(val):
    try:
        return pd.to_datetime(val).strftime('%H:%M:%S')
    except:
        return np.nan

if 'event_ts' in df.columns:
    df['event_ts'] = df['event_ts'].astype(str).str.strip()
    df['event_ts'] = df['event_ts'].apply(clean_event_time)

# Step 10: Final clean-up
df = df.drop_duplicates().reset_index(drop=True)

# Confirm
print("Data cleaned successfully!")  # Preview
# Final clean-up
df = df.drop_duplicates().reset_index(drop=True)

# Export to CSV
df.to_csv("cleaned_amex_offers_data.csv", index=False)

# print("Data cleaned and saved to 'cleaned_amex_offers_data.csv' successfully!")

