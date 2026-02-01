from pathlib import Path
import pandas as pd

# ====== 配置 ======
in_path = Path("aligned_fast_filled.csv")
out_path = in_path.with_name(in_path.stem + "_cleaned.csv")

# 需要删除的列（存在才删，不存在就跳过）
cols_to_drop = [
    "xia_dish name",
    "xia_description",
    "_match_score",
    "_matched_json_key",
    "json_key",
]

# “菜名”列候选名（按实际存在的那个来）
dish_name_candidates = [
    "菜名",
]

image_col = "图片地址"

# ====== 读取 ======
df = pd.read_csv(in_path, dtype=str)  # 全部按字符串读，避免空值/数字混乱

# ====== 删除“图片地址”为空的记录 ======
if image_col not in df.columns:
    raise KeyError(f"找不到列：{image_col}。当前列名：{list(df.columns)}")

img = df[image_col]
mask_empty_img = img.isna() | img.astype(str).str.strip().eq("")
df = df.loc[~mask_empty_img].copy()

# ====== 找到“菜名”列并去重（仅按菜名）=====
dish_col = next((c for c in dish_name_candidates if c in df.columns), None)
if dish_col is None:
    raise KeyError(
        f"找不到菜名列。候选：{dish_name_candidates}。当前列名：{list(df.columns)}"
    )

# 去重前也可先清理菜名两端空格
df[dish_col] = df[dish_col].astype(str).str.strip()

# 删除菜名为空的记录（可选；如不想删可注释掉这两行）
mask_empty_dish = df[dish_col].isna() | df[dish_col].str.strip().eq("")
df = df.loc[~mask_empty_dish].copy()

# 仅按菜名去重：保留第一次出现
df = df.drop_duplicates(subset=[dish_col], keep="first").copy()

# ====== 删除指定列 ======
existing_to_drop = [c for c in cols_to_drop if c in df.columns]
if existing_to_drop:
    df = df.drop(columns=existing_to_drop)

# ====== 保存 ======
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"完成：输出文件 -> {out_path}")
print(f"最终行数：{len(df)}")
print(f"使用去重列：{dish_col}")
print(f"已删除列：{existing_to_drop}")
