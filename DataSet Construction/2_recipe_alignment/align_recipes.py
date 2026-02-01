# -*- coding: utf-8 -*-
import json
import re
import time
from array import array
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ================== Configuration ==================
INPUT_CSV = "add_duplicate.csv"
JSON_FILE = "recipe_corpus_finetune.json"
OUTPUT_CSV = "aligned_fast.csv"
UNMATCHED_CSV = "unmatched_fast.csv"
UNUSED_JSON = "unused_fast.jsonl"
SIM_THRESHOLD = 80
MAX_RARE_GRAMS = 4
MAX_CANDIDATES = 5000
BUILD_LOG_EVERY = 200000
ALIGN_LOG_EVERY = 200
VERBOSE_MATCH = False

# ==========================================================
# Text normalization and Bigram generation for matching
# ==========================================================
_punct_re = re.compile(r"[\s\u3000`~!@#$%^&*()\-=+\[\]{}\\|;:'\",.<>/?，。！？；：、】【（）《》“”‘’、·…]+", re.UNICODE)

def normalize_text(s: Any) -> str:
    """Remove punctuation and spaces, convert to lowercase"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    s = _punct_re.sub("", s)
    return s.lower()

def make_bigrams(s: str) -> List[str]:
    """Split text into character pairs for indexing"""
    if not s:
        return []
    if len(s) == 1:
        return [s]
    return [s[i:i+2] for i in range(len(s) - 1)]

# ==========================================================
# Inverted Index building (Exact and Bigram index)
# ==========================================================
def load_json_records(path: str) -> List[Dict[str, Any]]:
    # ... logic to load JSON/JSONL records ...
    pass

def main():
    t0 = time.time()
    # Loading logic...
    
    # exact_map: normalized text -> indices
    exact_map: Dict[str, array] = defaultdict(lambda: array("I"))
    # bigram_index: character pairs -> indices
    bigram_index: Dict[str, array] = defaultdict(lambda: array("I"))

    # Build index for fast lookup
    for i, r in enumerate(valid_records):
        # ... index construction logic ...
        pass

    # ==========================================================
    # Alignment logic: Exact matching followed by Fuzzy matching
    # ==========================================================
    used_json_idx = set()
    matched_count = 0

    for row_idx in range(total_rows):
        raw_name = df.at[row_idx, "菜名"]
        qn = normalize_text(raw_name)

        # 1. Try Exact match
        # 2. Try Fuzzy match via Bigram recall and score ranking
        # ... (implementation details of pick_exact and pick_fuzzy) ...
        
        if match_idx is not None and score >= SIM_THRESHOLD:
            # Map external JSON data to CSV columns
            rec = valid_records[match_idx]
            df.at[row_idx, "xia_dish name"] = rec.get("dish", "")
            df.at[row_idx, "xia_recipeIngredient"] = rec.get("recipeIngredient", "")
            # ... additional assignments ...

    # ==========================================================
    # Output aligned data and unmatched remnants
    # ==========================================================
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    # ... logic for unmatched CSV and unused JSON ...

if __name__ == "__main__":
    main()