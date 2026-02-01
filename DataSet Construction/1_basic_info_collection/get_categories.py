import pandas as pd

# ==========================================================
# Configuration of file paths
# ==========================================================
CATEGORIES_FILE = 'processed_categories.csv'
OUTPUT_FILE = 'name_output.csv'
RESULT_FILE = 'name_output_with_categories.csv'


def get_main_category(category_name):
    """Get main category name (split by '\\' and take the first part)"""
    if pd.isna(category_name) or str(category_name).strip() == "":
        return "Uncategorized"
    s = str(category_name)
    return s.split('\\')[0] if '\\' in s else s


def main():
    # ==========================================================
    # Load datasets and validate required columns
    # ==========================================================
    # Read data
    categories_df = pd.read_csv(CATEGORIES_FILE)
    name_output_df = pd.read_csv(OUTPUT_FILE)

    # Keep only the category fields needed for merging and output
    cat_cols = ['子分类ID', '父分类ID', '分类名称']
    missing_cols = [c for c in cat_cols if c not in categories_df.columns]
    if missing_cols:
        raise ValueError(f"{CATEGORIES_FILE} is missing necessary columns: {missing_cols}")

    if '子分类ID' not in name_output_df.columns:
        raise ValueError(f"{OUTPUT_FILE} is missing necessary column: ['子分类ID']")

    # ==========================================================
    # Merge category information and extract main category
    # ==========================================================
    # Merge category names based on Subcategory ID
    merged_df = pd.merge(
        name_output_df,
        categories_df[cat_cols],
        on='子分类ID',
        how='left'
    )

    # Optional: Add a column for the main category
    merged_df['主分类'] = merged_df['分类名称'].apply(get_main_category)

    # ==========================================================
    # Save the final result and output statistics
    # ==========================================================
    # Save the result
    merged_df.to_csv(RESULT_FILE, index=False, encoding='utf-8-sig')

    # Output execution summary
    total = len(merged_df)
    missing_category = merged_df['分类名称'].isna().sum()
    print(f"Processing complete: {total} total records; {missing_category} records failed to match category names.")
    print(f"Result saved to: {RESULT_FILE}")


if __name__ == '__main__':
    main()