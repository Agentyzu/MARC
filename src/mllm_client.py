import hashlib
import json
import sqlite3
import re
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.config import MARCSettings

# =========================================================================
# MLLM Client & Caching System
# Corresponds to Paper Section 3.2: Multi-Agent Posterior Aggregation
# =========================================================================

class MARCClient:
    """
    Wraps Qwen2-VL model interaction and implements SQLite-based caching
    to avoid expensive re-inference costs.
    """
    def __init__(self, model_path: str, db_path: str):
        self.db_path = db_path
        print(f"Loading MLLM from {model_path}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=False
        )
        self._init_db()

    def _init_db(self):
        """Initializes the cache database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, val TEXT)")
        conn.close()

    def get_prediction(self, image_path: str, prompt: str, action_id: int, seed: int) -> dict:
        """
        Retrieves model prediction.
        Prioritizes cache lookup. If miss, performs inference.
        """
        # Generate unique cache key
        key_str = f"{image_path}_{action_id}_{prompt}_{seed}"
        key = hashlib.md5(key_str.encode()).hexdigest()

        # 1. Query Cache
        conn = sqlite3.connect(self.db_path)
        res = conn.execute("SELECT val FROM cache WHERE key=?", (key,)).fetchone()
        if res:
            conn.close()
            return json.loads(res[0])

        # 2. Perform Inference
        messages = [{
            "role": "user", 
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt",
            max_pixels=28 * 28 * 164  # VRAM optimization limit
        ).to(self.model.device)
        
        # Stochastic sampling for uncertainty estimation
        out_ids = self.model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=MARCSettings.TEMPERATURE
        )
        resp = self.processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)], 
            skip_special_tokens=True
        )[0]
        
        # 3. Parse Result
        try:
            # Robust JSON extraction
            json_str = re.search(r"\{.*?\}", resp, re.DOTALL).group(0)
            data = json.loads(json_str)
            result = {
                "delta": float(data.get('delta_kcal', 0.0)), 
                "lower": float(data.get('raw_lower', -50.0)), 
                "upper": float(data.get('raw_upper', 50.0))
            }
        except:
            # Safety fallback for parsing failures
            result = {"delta": 0.0, "lower": -100.0, "upper": 100.0}

        # 4. Write to Cache
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result)))
        conn.commit()
        conn.close()
        
        return result