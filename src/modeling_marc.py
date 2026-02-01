import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# =========================================================================
# MARC Modeling Components
# Corresponds to Paper Section 3.2: Lightweight Gated Calibration & Policy
# =========================================================================

class LGCNet(nn.Module):
    """
    Lightweight Gated Calibration (LGC) Mechanism.
    Corresponds to Paper Eq. (8): alpha_i = sigma(w^T z_i + b)
    Dynamically adjusts the residual scaling factor based on uncertainty.
    """
    def __init__(self, input_dim):
        super(LGCNet, self).__init__()
        # Input dim: sigma_delta (1) + y_base (1) + cuisine_embedding (9) = 11
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        return self.sigmoid(self.fc(z))

class PolicyRouter:
    """
    Contextual Prompt Routing Policy.
    Corresponds to Paper Section 3.2: Supervised Policy Learning.
    Uses Logistic Regression to select the best strategy based on features.
    """
    def __init__(self):
        # Use balanced class_weight to handle class imbalance
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)