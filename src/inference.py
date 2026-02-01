import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from src.config import MARCSettings
from src.train_lgc import get_mapa_stats

# =========================================================================
# Stage 3 & 4: Split Conformal Prediction & Final Evaluation
# Corresponds to Paper Section 3.2: Conformalized Residual Bounding
# =========================================================================

def calibrate_scp(data_provider, marc_client, lgc_net, policy_router, cal_df):
    """
    Stage 3: Calculate quantile q_hat on calibration set (Eq. 11).
    """
    print(">>> Stage 3: SCP Calibration...")
    
    X_cal = data_provider.get_router_features(cal_df)
    cal_acts = policy_router.predict(X_cal)
    
    eps_scores = []
    lgc_net.eval()
    
    for i, (idx, row) in enumerate(tqdm(cal_df.iterrows(), total=len(cal_df), desc="Calibrating")):
        dm, sig, samples = get_mapa_stats(row, cal_acts[i], marc_client)
        
        with torch.no_grad():
            z_tensor = data_provider.get_lgc_input_tensor(row, sig)
            alpha = lgc_net(z_tensor).item()
            
        # Initial uncalibrated interval (Eq. 9)
        L_i = row['y_base'] + alpha * np.quantile(samples, MARCSettings.BETA/2)
        U_i = row['y_base'] + alpha * np.quantile(samples, 1 - MARCSettings.BETA/2)
        
        # Nonconformity score calculation (Eq. 10)
        # max(Lower_Err, Upper_Err, 0)
        score = max(L_i - row['y_ref'], row['y_ref'] - U_i, 0)
        eps_scores.append(score)
        
    # Calculate q_hat (Finite-sample correction)
    n_cal = len(eps_scores)
    q_level = min(1.0, (n_cal + 1) * (1 - MARCSettings.ALPHA_PRIME) / n_cal)
    q_hat = np.quantile(eps_scores, q_level)
    
    print(f"Calibration complete. q_hat = {q_hat:.4f}")
    return q_hat

def evaluate_marc(data_provider, marc_client, lgc_net, policy_router, test_df, q_hat):
    """
    Stage 4: Evaluate MAE and Interval Coverage on test set.
    """
    print(">>> Stage 4: Final Evaluation...")
    
    X_test = data_provider.get_router_features(test_df)
    test_acts = policy_router.predict(X_test)
    
    results = []
    lgc_net.eval()
    
    for i, (idx, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Testing")):
        dm, sig, samples = get_mapa_stats(row, test_acts[i], marc_client)
        
        with torch.no_grad():
            z_tensor = data_provider.get_lgc_input_tensor(row, sig)
            alpha = lgc_net(z_tensor).item()
        
        # Final point estimate
        y_pred = row['y_base'] + alpha * dm
        
        # Initial interval
        L_pre = row['y_base'] + alpha * np.quantile(samples, MARCSettings.BETA/2)
        U_pre = row['y_base'] + alpha * np.quantile(samples, 1 - MARCSettings.BETA/2)
        
        # Calibrated interval (Eq. 12)
        L_star = L_pre - q_hat
        U_star = U_pre + q_hat
        
        results.append({
            'y_ref': row['y_ref'], 
            'y_pred': y_pred,
            'covered': (row['y_ref'] >= L_star and row['y_ref'] <= U_star),
            'width': U_star - L_star
        })
        
    res_df = pd.DataFrame(results)
    
    mae = mean_absolute_error(res_df['y_ref'], res_df['y_pred'])
    coverage = res_df['covered'].mean()
    mpiw = res_df['width'].mean()
    
    print("\n" + "="*40)
    print(f"MARC Performance on Test Set (N={len(test_df)})")
    print(f"MAE      : {mae:.2f} kcal")
    print(f"Coverage : {coverage:.1%} (Target: {1-MARCSettings.ALPHA_PRIME:.0%})")
    print(f"MPIW     : {mpiw:.2f} kcal")
    print("="*40)