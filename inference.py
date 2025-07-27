import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#  Load quantized parameters
quantized = joblib.load("quant_params.joblib")

# Dequantize weights and bias
weights = quantized['weights'].astype(float) / quantized['scale_w'] + quantized['min_w']
bias = quantized['bias'].astype(float) / quantized['scale_b'] + quantized['min_b']

# Convert to torch tensors
weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)  # shape [1, num_features]
bias_tensor = torch.tensor(bias, dtype=torch.float32)

# Define PyTorch model
class QuantizedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

#  Initialize and inject dequantized weights
input_dim = weights.shape[0]
model = QuantizedModel(input_dim)
with torch.no_grad():
    model.linear.weight.copy_(weights_tensor)
    model.linear.bias.copy_(bias_tensor)

# Load California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Run inference
# Convert X_test to torch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Ensure input tensor does NOT require gradients
X_test_tensor.requires_grad_(False)

# Set model to evaluation mode
model.eval()

# Clean inference block
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_np = y_pred.squeeze().detach().cpu().numpy()

# Evaluate
r2 = r2_score(y_test, y_pred)
print(f"R² Score of Quantized Model: {r2:.4f}")

# Check weight range before and after quantization
print("Original weights:", original_params['weights'])
print("Quantized weights:", quant_params['weights'])
print("Intercepts:", quant_params['intercept'])

# Plot comparison
import matplotlib.pyplot as plt
plt.scatter(y_true, y_pred)
plt.title("Predicted vs True Values")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.grid(True)
plt.show()
