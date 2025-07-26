import joblib
import os

# Load scikit-learn model
model_path = "model.joblib"  # Update this if your file is elsewhere
model = joblib.load(model_path)

# Extract parameters
params = {
    'weights': model.coef_,
    'bias': model.intercept_
}

#  Save unquantized parameters
joblib.dump(params, "unquant_params.joblib")
#  Verify save
print("Unquantized parameters saved to unquant_params.joblib")
print(f"File size: {os.path.getsize('unquant_params.joblib') / 1024:.2f} KB")
import numpy as np

def quantize(x):
    min_val, max_val = np.min(x), np.max(x)
    if max_val - min_val == 0:
        # All values are the same — return zeros and safe scale
        x_q = np.zeros_like(x, dtype=np.uint8)
        scale = 1.0  # Avoid divide-by-zero
    else:
        scale = 255 / (max_val - min_val)
        x_q = ((x - min_val) * scale).astype(np.uint8)
    return x_q, scale, min_val
joblib.dump(quantize, "quant_params.joblib")

