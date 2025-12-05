# src/feature_importance.py
import os, joblib, numpy as np
import matplotlib.pyplot as plt
from train import load_data

os.makedirs("reports", exist_ok=True)
obj = joblib.load("models/handwriting_mood_rf.pkl")
clf = obj['model']
X, y, classes = load_data("data/raw")
imp = clf.feature_importances_

# Save numeric importances
np.savetxt("reports/feature_importances.txt", imp, fmt="%.6f")
print("Saved reports/feature_importances.txt")

# Plot
plt.figure(figsize=(8,4))
plt.bar(range(len(imp)), imp)
plt.xlabel("Feature index")
plt.ylabel("Importance")
plt.title("RandomForest Feature Importances")
plt.tight_layout()
out = "reports/feature_importances.png"
plt.savefig(out)
print("Saved", out)
