from tqdm import tqdm
from src.config import MARCSettings
from src.prompt_utils import build_prompt

# =========================================================================
# Stage 1: Policy Routing Training
# Corresponds to Paper Eq. (2) & (3): Oracle Action Selection
# =========================================================================

def train_policy_router(data_provider, marc_client, policy_router, train_df):
    """
    1. Generate Oracle Actions (Offline calculation of best strategy)
    2. Train supervised policy network (Logistic Regression)
    """
    print(">>> Stage 1: Policy Routing & Oracle Generation...")
    
    oracle_labels = []
    
    # Iterate through training set to find Oracle via exhaustive search
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Oracle Calc"):
        best_reward = -float('inf')
        best_action = 6 # Default to Anchor
        
        for action in range(1, 7): # Actions a1 to a6
            # Get single inference result for Oracle evaluation
            res = marc_client.get_prediction(
                row['image_path'], 
                build_prompt(row, action), 
                action, 
                seed=42
            )
            
            y_hat = row['y_base'] + res['delta']
            error = abs(row['y_ref'] - y_hat)
            
            # Check if Ground Truth falls within predicted interval
            is_covered = 1.0 if (row['y_ref'] >= (row['y_base'] + res['lower']) and 
                                 row['y_ref'] <= (row['y_base'] + res['upper'])) else 0.0
            
            # Eq. (2): Action-Quality Reward
            reward = -MARCSettings.LAMBDA1 * error + MARCSettings.LAMBDA2 * is_covered
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
                
        oracle_labels.append(best_action)
    
    # Save Oracle labels for supervised learning
    train_df['oracle_act'] = oracle_labels
    
    # Extract multimodal features and fit classifier
    data_provider.fit_feature_extractors(train_df)
    X_train = data_provider.get_router_features(train_df)
    
    print("Fitting Policy Router...")
    policy_router.fit(X_train, train_df['oracle_act'])
    
    return policy_router, X_train # Return feature matrix for later use
