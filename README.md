# MARC: Beyond Visual Regression for Robust Calorie Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

This repository contains the official implementation of the paper **"Beyond Visual Regression: Anchoring Multimodal Residual Correction with Domain Priors for Robust Calorie Estimation"**, submitted to **IJCAI 2025**.

## 🌟 Introduction

**MARC** reformulates calorie estimation from a direct regression task into a **conditional residual prediction** task. By grounding inference on a retrieval-based USDA anchor, MARC adaptively selects reasoning strategies to correct visual deviations.

<div align="center">
 <img src="MARC_framework.png" alt="DynaSRL Architecture" width="800"/>
</div>

### Key Features
*   **Anchored Residual Learning:** Instead of predicting absolute calories, MARC predicts the deviation ($\Delta$) from a retrieved nutritional baseline.
*   **Contextual Prompt Routing:** A policy network selects the optimal reasoning strategy for different dish types.
*   **Multi-Agent Posterior Aggregation (MAPA):** Mitigates MLLM hallucinations by aggregating local beliefs from an ensemble of stochastic forward passes.
*   **Reliable Quantification:** Integrates Lightweight Gated Calibration (LGC) and Split Conformal Prediction (SCP) to provide statistically valid confidence intervals.

## 📁 Project Structure

```text
.
├── DataSet Construction/       # Multi-stage data pipeline
│   ├── 1_basic_info_collection/
│   │   ├── shop_url_id.py      # Scrape restaurant URLs and category IDs
│   │   ├── url_unique.py       # Deduplicate shop entries
│   │   ├── meal_name_spider.py # Extract dish names and images
│   │   └── get_categories.py   # Map IDs to human-readable categories
│   ├── 2_recipe_alignment/
│   │   ├── add_col_duplicate.py# Prepare CSV structure for matching
│   │   ├── align_recipes.py    # Fuzzy match dishes with USDA/Recipe corpus
│   │   └── delete_column.py    # Clean temporary audit columns
│   ├── 3_nutritional_enrichment/
│   │    ├── llm_food.py        # Batch calculate baseline nutrients
│   │    └── promptTemplate.py  # Template for LLM ingredient decomposition
│   └── data/
│       └── final.json          # Final enriched dataset
├── MARC/                       # Core algorithm implementation
│   ├── MARC.py                 # Main pipeline
│   ├── delta_cache.sqlite      # Persistent cache for MLLM outputs
│   └── Daud_selected_3000.csv  # Expert-verified Audited Subset (CCM-Gold)
└── requirements.txt            # Project dependencies
```

## ⚙️ Setup & Installation

The implementation is optimized for two **NVIDIA RTX PRO 6000 (96GB) GPUs** running **Ubuntu 20.04** with **Python 3.10** and **PyTorch 2.1.2**, and uses **Qwen2-VL** as the backbone MLLM. 

1. **Create Environment**
    ```bash
    conda create -n marc python=3.10
    conda activate marc
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

2. **Model Preparation**
    Place your base MLLM weights in the cache directory:
    ```bash
    # Default path used in MARC.py
    mkdir -p model_cache/qwen/Qwen2-VL-7B-Instruct
    ```

## 🚀 Usage

### 1. CCM Dataset Construction
Before running the MARC pipeline, you must construct the grounded dataset:
```bash
# Step 1: Collect raw shop data
python DataSet Construction/1_basic_info_collection/shop_url_id.py
python DataSet Construction/1_basic_info_collection/url_unique.py
python DataSet Construction/1_basic_info_collection/meal_name_spider.py
python DataSet Construction/1_basic_info_collection/get_categories.py

# Step 2: Align with recipe corpus
python DataSet Construction/2_recipe_alignment/add_col_duplicate.py
python DataSet Construction/2_recipe_alignment/align_recipes.py
python DataSet Construction/2_recipe_alignment/delete_column.py

# Step 3: Generate USDA Anchors (y_base)
python DataSet Construction/3_nutritional_enrichment/llm_food.py
```

### 2. Running the MARC Pipeline
Running the MARC pipeline:

```bash
python MARC/MARC.py
```

## 🔬 Reproducibility

*   **Code:** All code used for data collection, MARC pipeline, and evaluation is included. 
*   **CCM Dataset:** The China Culinary Multimodal (`final.json`) contains 32,671 dishes. 
*   **Hyperparameters:**  Key hyperparameters are documented in the python scripts and the paper appendix. 

## 🖋️ Citation

If you use MARC or the CCM dataset in your research, please cite:

```bibtex
@inproceedings{marc2025,
  title={Beyond Visual Regression: Anchoring Multimodal Residual Correction with Domain Priors for Robust Calorie Estimation},
  author={Anonymous},
  booktitle={Under Review},
  year={2026}
}

```


