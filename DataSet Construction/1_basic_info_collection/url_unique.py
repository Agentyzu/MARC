import pandas as pd

# ==========================================================
# Load raw data and remove duplicate URLs using Pandas
# ==========================================================

# Read CSV file
df = pd.read_csv('../shop_url_id/urls_output.csv')

# Remove duplicate URLs (keep the first occurrence)
df_unique = df.drop_duplicates(subset=['店铺URL'], keep='first')

# Save to new file
df_unique.to_csv('unique_urls.csv', index=False)