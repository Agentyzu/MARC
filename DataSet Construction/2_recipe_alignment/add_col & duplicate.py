import pandas as pd

INPUT_FILE = "name_output_with_categories.csv"
OUTPUT_DEDUP_FILE = "add_duplicate.csv"
OUTPUT_DUPLICATES_FILE = "duplicates.csv"

NEW_COLS = [
    "xia_dish name",
    "xia_recipeIngredient",
    "xia_description",
    "xia_recipeInstructions",
]

def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # ==========================================================
    # Add placeholder columns for upcoming recipe data
    # ==========================================================
    for c in NEW_COLS:
        df[c] = ""

    # ==========================================================
    # Identify and split duplicates based on the "Dish Name" column
    # ==========================================================
    if "菜名" not in df.columns:
        raise ValueError("Cannot find '菜名' (Dish Name) column in input file")

    # Treat missing values as identical strings
    key = df["菜名"].fillna("")

    # keep=first: mark first occurrence as original, others as duplicates
    dup_mask = key.duplicated(keep="first")

    # Duplicate records (to be removed)
    df_duplicates = df.loc[dup_mask].copy()

    # Unique records (to be kept)
    df_dedup = df.loc[~dup_mask].copy()

    # ==========================================================
    # Save processed files
    # ==========================================================
    df_dedup.to_csv(OUTPUT_DEDUP_FILE, index=False, encoding="utf-8-sig")
    df_duplicates.to_csv(OUTPUT_DUPLICATES_FILE, index=False, encoding="utf-8-sig")

    print(f"After De-duplication: {len(df_dedup)} rows -> {OUTPUT_DEDUP_FILE}")
    print(f"Duplicate data: {len(df_duplicates)} rows -> {OUTPUT_DUPLICATES_FILE}")

if __name__ == "__main__":
    main()