# plot_tree.py
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import os

model_path = "app/models/cupid_match_model_best.pkl"
save_path = "app/static/tree_plot.png"

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Plot the first tree
xgb.plot_tree(model, num_trees=0)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Tree saved to {save_path}")
