import joblib
import numpy as np
from sklearn.metrics import r2_score

# Load the test dataset
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Load the model
model = joblib.load("model/sklearn_model.joblib")

# Run prediction
y_pred = model.predict(X_test)

# Print evaluation score
score = r2_score(y_test, y_pred)
print(f"R² Score: {score:.4f}")
