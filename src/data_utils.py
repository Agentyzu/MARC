import os
import hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from src.config import MARCSettings

# =========================================================================
# Data Loading and Feature Extraction
# Corresponds to Paper Section 3.1: Data Preprocessing
# =========================================================================

class MARCDataProvider:
    def __init__(self, data_path: str):
        if data_path.endswith('.json'):
            self.df = pd.read_json(data_path)
        else:
            self.df = pd.read_csv(data_path)
            
        self._init_preprocessing()
        
        # Initialize feature extractors
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
        # Load CLIP model for visual embeddings (for Policy Routing)
        print("Loading CLIP model for visual embedding...")
        # STRICT MATCH: Added use_safetensors=True as per original MARC.py
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_safetensors=True
        ).to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Cuisine encoding mapping (Strictly matching MARC.py)
        self.cuisines = ['Jiangzhe', 'Guangdong', 'Hunan', 'Shandong', 'Dongbei', 'Sichuan', 'Anhui', 'Beijing', 'Other']
        self.c_to_idx = {c: i for i, c in enumerate(self.cuisines)}

    def _init_preprocessing(self):
        """Handles image paths and basic columns."""
        os.makedirs(MARCSettings.IMAGE_CACHE_DIR, exist_ok=True)
        
        def get_actual_path(url):
            bn = hashlib.md5(str(url).encode()).hexdigest()
            for ext in ['.webp', '.jpg', '.jpeg', '.png']:
                p = os.path.join(MARCSettings.IMAGE_CACHE_DIR, f"{bn}{ext}")
                if os.path.exists(p): return p
            return os.path.join(MARCSettings.IMAGE_CACHE_DIR, f"{bn}.webp")

        if 'y_base' not in self.df.columns: 
            self.df['y_base'] = self.df['calorie'] 
        
        self.df['image_path'] = self.df['图片地址'].apply(get_actual_path)

    def get_splits(self):
        """
        Splits dataset according to the paper: 
        D_train (2000), D_cal (500), D_test (500)
        """
        train_df = self.df.iloc[:2000].copy()
        cal_df = self.df.iloc[2000:2500].copy()
        test_df = self.df.iloc[2500:3000].copy()
        return train_df, cal_df, test_df

    def get_visual_embedding(self, img_path):
        """Extracts CLIP visual features."""
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                return self.clip_model.get_image_features(**inputs).cpu().numpy().flatten()
        except: 
            return np.zeros(512)

    def fit_feature_extractors(self, train_df):
        """Fits textual TF-IDF and numerical Scaler."""
        text_data = train_df['菜名'] + " " + train_df['xia_recipeIngredient'].fillna('')
        self.vectorizer.fit(text_data)
        self.scaler.fit(train_df[['y_base']].values)

    def get_router_features(self, df):
        """
        Constructs input features for Policy Router:
        Concatenate(Text_Embedding, Visual_Embedding, Baseline_Calories)
        """
        T = self.vectorizer.transform(df['菜名'] + " " + df['xia_recipeIngredient'].fillna(''))
        V = np.array([self.get_visual_embedding(p) for p in df['image_path']])
        B = self.scaler.transform(df[['y_base']].values)
        return hstack([T, V, B])

    def get_lgc_input_tensor(self, row, sigma_delta):
        """
        Constructs input tensor z_i for LGC network:
        [sigma_delta, y_base, one_hot_cuisine]
        """
        e_cuis = np.zeros(MARCSettings.CUISINE_DIM)
        c_name = row.get('cuisine', 'Other')
        e_cuis[self.c_to_idx.get(c_name, 8)] = 1.0
        
        z = [float(sigma_delta), float(row['y_base'])] + list(e_cuis)
        return torch.tensor(z, dtype=torch.float32).to("cuda")