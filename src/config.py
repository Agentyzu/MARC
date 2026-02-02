# -*- coding: utf-8 -*-
# =========================================================================
# MARC Configuration
# Corresponds to Paper Section 4.1: Experimental Setup
# =========================================================================

class MARCSettings:
    """
    Stores all hyperparameter configurations for the MARC framework.
    """
    # Optimization Objective Weights (Eq. 2)
    LAMBDA1 = 1.0        # Weight for absolute error penalty
    LAMBDA2 = 100.0      # Weight for interval coverage reward (Best trade-off)

    # Sampling Parameters (Section 4.4 Analysis)
    J_AGENTS = 3         # Number of MLLM agents (Ensemble size)
    K_SAMPLES = 5        # Number of stochastic samples per agent
    TEMPERATURE = 0.7    # Sampling temperature (Optimal "sweet spot")

    # Conformal Prediction Parameters (Section 3.2)
    BETA = 0.1           # Initial quantile range (90% interval)
    ALPHA_PRIME = 0.1    # Target miscoverage level

    # Dimension Definitions
    # Table 1: Jiangzhe, Guangdong, Hunan, Shandong, Dongbei, Sichuan, Anhui, Beijing, Other
    CUISINE_DIM = 9      
    
    # LGC Training Parameters
    LGC_BATCH_SIZE = 64  
    LGC_EPOCHS = 50      
    LGC_LR = 0.01

    # File Paths
    MODEL_PATH = "model_cache/qwen/Qwen2-VL-7B-Instruct"
    DATA_PATH = "data/input/Daud_selected_3000.json"
    
    DB_PATH = "data/output/delta_cache.sqlite"
    IMAGE_CACHE_DIR = "image_cache"
