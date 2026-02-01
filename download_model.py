# -*- coding: utf-8 -*-
import os
from huggingface_hub import snapshot_download
from src.config import MARCSettings

# =========================================================================
# Model Download Script
# Downloads Qwen2-VL weights to the specified cache directory
# =========================================================================

def download_qwen_model():
    """
    Downloads the Qwen2-VL-7B-Instruct model from Hugging Face.
    The local path is determined by MARCSettings.MODEL_PATH.
    """
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    local_dir = MARCSettings.MODEL_PATH
    
    print(f"Starting download for {model_id}...")
    print(f"Target directory: {local_dir}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True
        )
        print(">>> Model downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_qwen_model()