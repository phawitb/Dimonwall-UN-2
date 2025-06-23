import pandas as pd
import matplotlib.pyplot as plt
import os

# Load training log
df = pd.read_csv("output/training_log.csv")

# Prepare output plot directory
PLOT_DIR = "output/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Reshape for plotting
metrics = ["RMSE", "MAE", "R2"]
melted = pd.melt(
    df,
    id_vars=["Target", "Model"],
    value_vars=[f"{t}_{m}" for m in metrics for t in ["Train", "Test"]],
    var_name="Metric",
    value_name="Score"
)

# Plot and save each metric for each target
for metric in metrics:
    for target in melted["Target"].unique():
        subset = melted[(melted["Target"] == target) & (melted["Metric"].str.contains(metric))]
        pivot = subset.pivot(index="Model", columns="Metric", values="Score")

        # Filter only Train/Test columns for this metric
        cols = [f"Train_{metric}", f"Test_{metric}"]
        pivot = pivot[cols]

        # Plot
        ax = pivot.plot(kind="bar", figsize=(10, 6), title=f"{target} - {metric}")
        plt.ylabel(metric)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save as PNG
        filename = f"{target.replace(' ', '_')}_{metric}.png".replace(":", "_")
        plt.savefig(os.path.join(PLOT_DIR, filename))
        # plt.show()
