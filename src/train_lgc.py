import numpy as np
import torch
import torch.optim as nn_optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.config import MARCSettings
from src.prompt_utils import build_prompt

# =========================================================================
# Stage 2: LGC Network Training
# Corresponds to Paper Section 3.2: Uncertainty-Aware Probabilistic Refinement
# =========================================================================

def get_mapa_stats(row, action_id, marc_client):
    """
    Multi-Agent Posterior Aggregation (MAPA).
    Obtains mean (delta) and uncertainty (sigma) from MLLM ensemble.
    """
    agent_means = []
    agent_vars = []
    all_samples = []
    
    # J agents, each sampling K times
    for j in range(MARCSettings.J_AGENTS):
        samples = []
        for k in range(MARCSettings.K_SAMPLES):
            # Simulate different paths by changing seed
            res = marc_client.get_prediction(
                row['image_path'], 
                build_prompt(row, action_id), 
                action_id, 
                seed=k + j*100
            )
            samples.append(res['delta'])
            
        all_samples.extend(samples)
        agent_means.append(np.mean(samples))
        # Add epsilon to prevent division by zero
        agent_vars.append(np.var(samples) + 1e-6)
        
    # GMM Posterior Aggregation (Eq. 7)
    delta_mllm = np.mean(agent_means)
    
    # Mixture Variance calculation
    term1 = np.mean(agent_vars)
    term2 = np.mean([m**2 for m in agent_means])
    term3 = delta_mllm**2
    sigma_delta = np.sqrt(max(0, term1 + term2 - term3))
    
    return delta_mllm, sigma_delta, all_samples

def train_lgc_net(data_provider, marc_client, lgc_net, policy_router, train_df, X_train):
    """
    Trains the LGC network to predict the optimal scaling factor alpha.
    """
    print(">>> Stage 2: LGC Training...")
    
    # 1. Predict strategies for current dataset using trained Router
    train_acts = policy_router.predict(X_train)
    
    # 2. Collect LGC training data
    z_list, d_list, t_list = [], [], []
    
    for i, (idx, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df), desc="LGC Data Prep")):
        dm, sig, _ = get_mapa_stats(row, train_acts[i], marc_client)
        
        z_tensor = data_provider.get_lgc_input_tensor(row, sig)
        z_list.append(z_tensor)
        d_list.append(dm) # Raw MLLM residual
        t_list.append(row['y_ref'] - row['y_base']) # Ground Truth residual
        
    # 3. Construct PyTorch DataLoader
    dataset = TensorDataset(
        torch.stack(z_list).float(), 
        torch.tensor(d_list, dtype=torch.float32).to("cuda"), 
        torch.tensor(t_list, dtype=torch.float32).to("cuda")
    )
    loader = DataLoader(dataset, batch_size=MARCSettings.LGC_BATCH_SIZE, shuffle=True)
    
    # 4. Training Loop
    optimizer = nn_optim.Adam(lgc_net.parameters(), lr=MARCSettings.LGC_LR)
    loss_fn = nn.HuberLoss() # Robust loss per paper
    
    lgc_net.train()
    for epoch in range(MARCSettings.LGC_EPOCHS):
        epoch_loss = 0
        for b_z, b_d, b_t in loader:
            optimizer.zero_grad()
            alpha = lgc_net(b_z).squeeze()
            
            # Gated Correction: alpha * raw_residual
            gated_correction = alpha * b_d
            
            loss = loss_fn(gated_correction, b_t.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{MARCSettings.LGC_EPOCHS}, Loss: {epoch_loss/len(loader):.4f}")

    return lgc_net