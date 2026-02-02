from src.config import MARCSettings
from src.data_utils import MARCDataProvider
from src.mllm_client import MARCClient
from src.modeling_marc import LGCNet, PolicyRouter
from src.train_policy import train_policy_router
from src.train_lgc import train_lgc_net
from src.inference import calibrate_scp, evaluate_marc

# =========================================================================
# MARC Pipeline Entry Point
# Executes Phases 1-4 Sequentially
# =========================================================================

def main():
    # 1. Initialize basic components
    print("Initializing MARC components...")
    data_provider = MARCDataProvider(MARCSettings.DATA_PATH)
    
    data_provider.prepare_all_images() 
    
    marc_client = MARCClient(
        model_path=MARCSettings.MODEL_PATH, 
        db_path=MARCSettings.DB_PATH
    )
    
    policy_router = PolicyRouter()
    
    # Dims: sigma(1) + ybase(1) + cuisine(9)
    lgc_net = LGCNet(input_dim=2 + MARCSettings.CUISINE_DIM).to("cuda")

    # 2. Data splitting
    train_df, cal_df, test_df = data_provider.get_splits()
    print(f"Data Splits: Train={len(train_df)}, Cal={len(cal_df)}, Test={len(test_df)}")

    # 3. Stage 1: Train Policy Network
    policy_router, X_train = train_policy_router(
        data_provider, marc_client, policy_router, train_df
    )

    # 4. Stage 2: Train LGC Network
    lgc_net = train_lgc_net(
        data_provider, marc_client, lgc_net, policy_router, train_df, X_train
    )

    # 5. Stage 3: Conformal Prediction Calibration (Get q_hat)
    q_hat = calibrate_scp(
        data_provider, marc_client, lgc_net, policy_router, cal_df
    )

    # 6. Stage 4: Final Evaluation
    evaluate_marc(
        data_provider, marc_client, lgc_net, policy_router, test_df, q_hat
    )

if __name__ == "__main__":
    main()
