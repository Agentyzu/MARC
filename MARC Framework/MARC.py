# -*- coding: utf-8 -*-
import os
import re
import json
import sqlite3
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, CLIPModel, CLIPProcessor
from qwen_vl_utils import process_vision_info

# ==========================================================
# 1. Hyperparameter Settings
# ==========================================================
class MARCSettings:
    # Reward balancing weights for Oracle selection
    LAMBDA1 = 1.0        # Weight for point accuracy (MAE)
    LAMBDA2 = 100.0      # Weight for interval validity (Coverage)
    
    # Uncertainty Estimation parameters 
    J_AGENTS = 3         # Number of independent MLLM ensemble agents
    K_SAMPLES = 5        # Stochastic forward passes per agent
    TEMPERATURE = 0.7    # Sampling temperature for reasoning diversity
    
    # Calibration and Quantile parameters
    BETA = 0.1           # Significance level for raw intervals
    ALPHA_PRIME = 0.1    # Target miscoverage level for Conformal Prediction
    
    # Feature dimensions and training configs
    CUISINE_DIM = 9      # Dimension for one-hot encoded cuisine types
    LGC_BATCH_SIZE = 64  
    LGC_EPOCHS = 50      

# ==========================================================
# 2. Lightweight Gated Calibration (LGC) Network
# ==========================================================
class LGCNet(nn.Module):
    def __init__(self, input_dim):
        super(LGCNet, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # Outputs gating coefficient alpha in (0, 1)
        return self.sigmoid(self.fc(z))

# ==========================================================
# 3. MLLM Client (Interface for Qwen2-VL)
# ==========================================================
class MARCClient:
    def __init__(self, model_path: str, db_path: str):
        self.db_path = db_path
        # Load MLLM with auto device mapping for efficiency
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=False # Ensure compatibility with specific checkpoints
        )
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database to cache MLLM responses to save compute."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, val TEXT)")
        conn.close()

    def get_prediction(self, image_path: str, prompt: str, action_id: int, seed: int) -> Dict:
        """Fetch MLLM prediction with local caching logic."""
        key = hashlib.md5(f"{image_path}_{action_id}_{prompt}_{seed}".encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        res = conn.execute("SELECT val FROM cache WHERE key=?", (key,)).fetchone()
        if res:
            conn.close()
            return json.loads(res[0])

        # Prepare multimodal inputs
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        # Pixel constraints to prevent OOM
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt",
            max_pixels=28 * 28 * 164 
        ).to(self.model.device)
        
        # Generate with stochastic decoding
        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=MARCSettings.TEMPERATURE)
        resp = self.processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out_ids)], skip_special_tokens=True)[0]
        
        # Extract numerical results from JSON response
        try:
            json_str = re.search(r"\{.*?\}", resp, re.DOTALL).group(0)
            data = json.loads(json_str)
            result = {
                "delta": float(data.get('delta_kcal', 0.0)), 
                "lower": float(data.get('raw_lower', -50.0)), 
                "upper": float(data.get('raw_upper', 50.0))
            }
        except:
            # Fallback for parsing errors
            result = {"delta": 0.0, "lower": -100.0, "upper": 100.0}

        conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result)))
        conn.commit()
        conn.close()
        return result

# ==========================================================
# 4. MARC Main Application Pipeline
# ==========================================================
class MARCApp:
    def __init__(self, model_path, csv_path, db_path):
        os.makedirs("image_cache", exist_ok=True)
        self.client = MARCClient(model_path, db_path)
        self.df = pd.read_csv(csv_path)
        
        # Load CLIP for fixed-size visual embeddings used in the Policy Router
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_safetensors=True 
        ).to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Map for one-hot encoding of Chinese cuisines
        self.cuisines = ['Jiangzhe', 'Guangdong', 'Hunan', 'Shandong', 'Dongbei', 'Sichuan', 'Anhui', 'Beijing', 'Other']
        self.c_to_idx = {c: i for i, c in enumerate(self.cuisines)}

        # Helper to locate images in local cache
        def get_actual_path(url):
            bn = hashlib.md5(str(url).encode()).hexdigest()
            for ext in ['.webp', '.jpg', '.jpeg', '.png']:
                p = f"image_cache/{bn}{ext}"
                if os.path.exists(p): return p
            return f"image_cache/{bn}.webp"

        # Baseline calorie setup (USDA Anchor)
        if 'y_base' not in self.df.columns: self.df['y_base'] = self.df['calorie']
        self.df['image_path'] = self.df['图片地址'].apply(get_actual_path)
        
        # Dataset split according to paper settings
        self.train_df = self.df.iloc[:2000].copy()
        self.cal_df = self.df.iloc[2000:2500].copy()
        self.test_df = self.df.iloc[2500:3000].copy()
        
        # Router components
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.router = LogisticRegression(max_iter=1000, class_weight='balanced')
        
        # LGC and SCP state
        self.lgc_net = LGCNet(input_dim=2 + MARCSettings.CUISINE_DIM).to("cuda")
        self.q_hat = 0.0

    def _get_visual_embedding(self, img_path):
        """Extract visual features using CLIP encoder."""
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                return self.clip_model.get_image_features(**inputs).cpu().numpy().flatten()
        except: return np.zeros(512)

    def _get_z_feature(self, row, sigma_delta):
        """Construct the input vector z for the LGC module."""
        e_cuis = np.zeros(MARCSettings.CUISINE_DIM)
        c_name = row.get('cuisine', 'Other')
        e_cuis[self.c_to_idx.get(c_name, 8)] = 1.0
        # z = [Uncertainty, Baseline, Cuisine_OneHot]
        z = [float(sigma_delta), float(row['y_base'])] + list(e_cuis)
        return torch.tensor(z, dtype=torch.float32).to("cuda")

    def _build_prompt(self, row, action_id):
        """Construct task-specific prompts for Contextual Prompt Routing."""
        STRATEGY_MAP = {
            1: "a1 (Visual Volumetric Scaling): Estimate served portion relative to recipe standard by analyzing visual cues like plate size and food volume.",
            2: "a2 (Implicit Additive Inference): Infer extra caloric contributions (e.g., hidden oil or sauce) by detecting textual cues in cooking instructions.",
            3: "a3 (Culinary-Guided Adjustment): Adjust for cooking technique absorption or moisture loss (e.g., oil absorption during deep-frying).",
            4: "a4 (Ingredient-Level Rectification): Check for missing or added high-calorie items (e.g., nuts, sugar, batter) that deviate from the standard recipe.",
            5: "a5 (Factorized Decomposition): Decompose the estimation into intermediate factors (portion, oil, ingredients), then derive the final residual Δ.",
            6: "a6 (Residual Anchoring): Anchoring the prediction to the baseline. Shrink Δ → 0 unless there is compelling multimodal evidence of deviation."
        }

        strategy_desc = STRATEGY_MAP.get(action_id, STRATEGY_MAP[6])

        # Core task instructions for MLLM
        prompt = (
            f"System: You are an expert dietitian. Your task is to estimate the calorie residual (Δ) of a dish.\n"
            f"Dish Name: {row['菜名']}\n"
            f"Standard Recipe Ingredients: {row['xia_recipeIngredient']}\n"
            f"Baseline Calories (USDA Anchor): {row['y_base']} kcal\n\n"
            f"Selected Reasoning Strategy: {strategy_desc}\n\n"
            f"Instruction: Based on the image and the strategy above, determine how much the actual dish's calories "
            f"deviate from the baseline. If the dish looks exactly like the standard recipe, Δ should be near 0.\n"
            f"Output strictly in JSON format:\n"
            f"{{\"delta_kcal\": <float>, \"raw_lower\": <float>, \"raw_upper\": <float>}}"
        )
        return prompt

    def get_mapa_stats(self, row, act):
        """Multi-Agent Posterior Aggregation (MAPA) to estimate residual and uncertainty."""
        agent_means, agent_vars = [], []
        all_samples = []
        for j in range(MARCSettings.J_AGENTS):
            # Stochastic sampling for each agent
            samples = [self.client.get_prediction(row['image_path'], self._build_prompt(row, act), act, seed=k)['delta'] 
                       for k in range(MARCSettings.K_SAMPLES)]
            all_samples.extend(samples)
            agent_means.append(np.mean(samples))
            agent_vars.append(np.var(samples) + 1e-6)
        
        # GMM Aggregation logic
        delta_mllm = np.mean(agent_means)
        # Mixture standard deviation accounts for both intra-agent and inter-agent variance
        sigma_delta = np.sqrt(max(0, np.mean(agent_vars) + np.mean([m**2 for m in agent_means]) - delta_mllm**2))
        return delta_mllm, sigma_delta, all_samples

    def run_pipeline(self):
        # --- Stage 1: Policy Routing (Offline Contextual Optimization) ---
        print(">>> Stage 1: Policy Routing & Oracle Generation...")
        oracle_labels = []
        for _, row in tqdm(self.train_df.iterrows(), total=2000, desc="Oracle Calc"):
            best_r, best_a = -float('inf'), 6
            # Exhaustive search for the best strategy (Oracle) on training set
            for a in range(1, 7):
                res = self.client.get_prediction(row['image_path'], self._build_prompt(row, a), a, seed=42)
                y_hat = row['y_base'] + res['delta']
                error = abs(row['y_ref'] - y_hat)
                # Reward function balances accuracy and coverage
                covered = 1.0 if (row['y_ref'] >= (row['y_base'] + res['lower']) and row['y_ref'] <= (row['y_base'] + res['upper'])) else 0.0
                reward = -MARCSettings.LAMBDA1 * error + MARCSettings.LAMBDA2 * covered
                if reward > best_r:
                    best_r, best_a = reward, a
            oracle_labels.append(best_a)
        
        # Train the Policy Router (Logistic Regression) using visual and text features
        self.train_df['oracle_act'] = oracle_labels
        self.vectorizer.fit(self.train_df['菜名'] + " " + self.train_df['xia_recipeIngredient'].fillna(''))
        self.scaler.fit(self.train_df[['y_base']].values)
        
        def get_router_feats(df):
            T = self.vectorizer.transform(df['菜名'] + " " + df['xia_recipeIngredient'].fillna(''))
            V = np.array([self._get_visual_embedding(p) for p in df['image_path']])
            B = self.scaler.transform(df[['y_base']].values)
            return hstack([T, V, B])

        X_train = get_router_feats(self.train_df)
        self.router.fit(X_train, self.train_df['oracle_act'])

        # --- Stage 2: LGC Training ---
        print(">>> Stage 2: LGC Training...")
        train_acts = self.router.predict(X_train)
        z_list, d_list, t_list = [], [], []
        for i, (idx, row) in enumerate(tqdm(self.train_df.iterrows(), total=2000, desc="LGC Data Prep")):
            dm, sig, _ = self.get_mapa_stats(row, train_acts[i])
            z_list.append(self._get_z_feature(row, sig))
            d_list.append(dm)
            t_list.append(row['y_ref'] - row['y_base']) # Target residual

        # Optimize LGC parameters using Huber Loss for robustness
        loader = DataLoader(TensorDataset(torch.stack(z_list), torch.tensor(d_list).to("cuda"), torch.tensor(t_list).to("cuda")), batch_size=64, shuffle=True)
        opt = optim.Adam(self.lgc_net.parameters(), lr=0.01)
        for epoch in range(MARCSettings.LGC_EPOCHS):
            for b_z, b_d, b_t in loader:
                opt.zero_grad()
                alpha = self.lgc_net(b_z).squeeze()
                loss = nn.HuberLoss()(alpha * b_d, b_t.float())
                loss.backward(); opt.step()

        # --- Stage 3: SCP Calibration (Split Conformal Prediction) ---
        print(">>> Stage 3: SCP Calibration...")
        X_cal = get_router_feats(self.cal_df)
        cal_acts = self.router.predict(X_cal)
        eps_scores = []
        for i, (idx, row) in enumerate(tqdm(self.cal_df.iterrows(), total=500, desc="SCP Scores")):
            dm, sig, samples = self.get_mapa_stats(row, cal_acts[i])
            with torch.no_grad():
                alpha = self.lgc_net(self._get_z_feature(row, sig)).item()
            # Calculate raw interval boundaries
            L_i = row['y_base'] + alpha * np.quantile(samples, MARCSettings.BETA/2)
            U_i = row['y_base'] + alpha * np.quantile(samples, 1 - MARCSettings.BETA/2)
            # Nonconformity score
            eps_scores.append(max(L_i - row['y_ref'], row['y_ref'] - U_i, 0))
        
        # Calculate q_hat for statistical coverage guarantee
        n_cal = len(eps_scores)
        self.q_hat = np.quantile(eps_scores, min(1.0, (n_cal + 1) * (1 - MARCSettings.ALPHA_PRIME) / n_cal))

        # --- Stage 4: Evaluation ---
        print(">>> Stage 4: Final Evaluation...")
        X_test = get_router_feats(self.test_df)
        test_acts = self.router.predict(X_test)
        results = []
        for i, (idx, row) in enumerate(tqdm(self.test_df.iterrows(), total=500, desc="Testing")):
            dm, sig, samples = self.get_mapa_stats(row, test_acts[i])
            with torch.no_grad():
                alpha = self.lgc_net(self._get_z_feature(row, sig)).item()
            
            # Final point prediction
            y_pred = row['y_base'] + alpha * dm
            # Preliminary interval
            L_pre = row['y_base'] + alpha * np.quantile(samples, MARCSettings.BETA/2)
            U_pre = row['y_base'] + alpha * np.quantile(samples, 1 - MARCSettings.BETA/2)
            # Final calibrated interval
            L_star, U_star = L_pre - self.q_hat, U_pre + self.q_hat
            
            results.append({
                'y_ref': row['y_ref'], 'y_pred': y_pred, 
                'covered': (row['y_ref'] >= L_star and row['y_ref'] <= U_star),
                'width': U_star - L_star
            })

        # Summary Metrics
        res_df = pd.DataFrame(results)
        print("\n" + "="*30)
        print(f"MAE: {mean_absolute_error(res_df['y_ref'], res_df['y_pred']):.2f} kcal")
        print(f"Coverage: {res_df['covered'].mean():.1%}")
        print(f"MPIW (Mean Prediction Interval Width): {res_df['width'].mean():.2f} kcal")
        print("="*30)

if __name__ == "__main__":
    # Execute the pipeline with local model paths
    app = MARCApp(
        "model_cache/qwen/Qwen2-VL-7B-Instruct", 
        "Daud_selected_3000_ref.csv", 
        "delta_cache.sqlite"
    )
    app.run_pipeline()