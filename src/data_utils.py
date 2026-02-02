import os
import hashlib
import requests
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from src.config import MARCSettings

# =========================================================================
# Data Provider and Preprocessing Utility
# Corresponds to Paper Section 3.1: Data Preprocessing
# Handles data loading, image caching, and feature extraction.
# =========================================================================

class MARCDataProvider:
    def __init__(self, data_path: str):
        """
        Initializes the provider by loading JSON/CSV and setting up paths.
        """
        # Load dataset based on file extension
        if data_path.endswith('.json'):
            self.df = pd.read_json(data_path)
        else:
            self.df = pd.read_csv(data_path)
            
        self._init_preprocessing()
        
        # Initialize NLP and Visual feature extractors
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
        # Load CLIP model (Strictly matching MARC.py with use_safetensors=True)
        print("Loading CLIP model for visual embedding...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_safetensors=True
        ).to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Mapping for regional cuisines
        self.cuisines = ['Jiangzhe', 'Guangdong', 'Hunan', 'Shandong', 'Dongbei', 'Sichuan', 'Anhui', 'Beijing', 'Other']
        self.c_to_idx = {c: i for i, c in enumerate(self.cuisines)}

    def _init_preprocessing(self):
        """Sets up the image cache directory and resolves baseline columns."""
        os.makedirs(MARCSettings.IMAGE_CACHE_DIR, exist_ok=True)
        
        if 'y_base' not in self.df.columns: 
            self.df['y_base'] = self.df['calorie'] 
        
        # Pre-resolve local paths based on URL MD5
        url_col = "图片地址" if "图片地址" in self.df.columns else "url"
        self.df['image_path'] = self.df[url_col].apply(self._get_local_path_from_url)

    def _get_local_path_from_url(self, url):
        """Generates a consistent local filename using MD5 hashing."""
        if pd.isna(url) or not str(url).startswith("http"):
            return None
        bn = hashlib.md5(str(url).encode()).hexdigest()
        # Default to webp/jpg resolution logic from original code
        for ext in ['.webp', '.jpg', '.jpeg', '.png']:
            p = os.path.join(MARCSettings.IMAGE_CACHE_DIR, f"{bn}{ext}")
            if os.path.exists(p): return p
        return os.path.join(MARCSettings.IMAGE_CACHE_DIR, f"{bn}.webp")

    # =========================================================================
    # Integrated Image Download Logic
    # =========================================================================

    def download_single_image(self, url, timeout=20, retries=3):
        """
        Downloads a single image from a URL and saves it to the local cache.
        """
        local_path = self._get_local_path_from_url(url)
        if not local_path: return None
        
        # Skip if already downloaded
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        headers = {"User-Agent": "Mozilla/5.0 MARC-Bot/1.0"}
        for _ in range(retries):
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    return local_path
            except:
                continue
        return None

    def prepare_all_images(self):
        """
        Batch downloads all images in the dataset. Useful for pre-caching.
        """
        url_col = "图片地址" if "图片地址" in self.df.columns else "url"
        print(f"Starting batch download for {len(self.df)} images...")
        
        success = 0
        for url in tqdm(self.df[url_col], desc="Downloading"):
            if self.download_single_image(url):
                success += 1
        print(f"Download complete. Successfully cached {success} images.")

    # =========================================================================
    # Feature Extraction and Splitting Logic
    # =========================================================================

    def get_splits(self):
        """Splits data into Train (2k), Cal (500), and Test (500)."""
        train_df = self.df.iloc[:2000].copy()
        cal_df = self.df.iloc[2000:2500].copy()
        test_df = self.df.iloc[2500:3000].copy()
        return train_df, cal_df, test_df

    def get_visual_embedding(self, img_path):
        """Extracts CLIP visual features from a local image."""
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                return self.clip_model.get_image_features(**inputs).cpu().numpy().flatten()
        except: 
            return np.zeros(512)

    def fit_feature_extractors(self, train_df):
        """Fits TF-IDF and Scaler on the training split."""
        text_data = train_df['菜名'] + " " + train_df['xia_recipeIngredient'].fillna('')
        self.vectorizer.fit(text_data)
        self.scaler.fit(train_df[['y_base']].values)

    def get_router_features(self, df):
        """Constructs concatenated features for the Policy Router."""
        T = self.vectorizer.transform(df['菜名'] + " " + df['xia_recipeIngredient'].fillna(''))
        V = np.array([self.get_visual_embedding(p) for p in df['image_path']])
        B = self.scaler.transform(df[['y_base']].values)
        return hstack([T, V, B])

    def get_lgc_input_tensor(self, row, sigma_delta):
        """Constructs the z_i input tensor for the LGC network."""
        e_cuis = np.zeros(MARCSettings.CUISINE_DIM)
        c_name = row.get('cuisine', 'Other')
        e_cuis[self.c_to_idx.get(c_name, 8)] = 1.0
        z = [float(sigma_delta), float(row['y_base'])] + list(e_cuis)
        return torch.tensor(z, dtype=torch.float32).to("cuda")

# =========================================================================
# CLI Entry Point for Data Preparation
# =========================================================================
if __name__ == "__main__":
    # If run directly, this script acts as the data_prep tool
    provider = MARCDataProvider(MARCSettings.DATA_PATH)
    provider.prepare_all_images()
