import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the models directory
models_dir = "models"

# Initialize a dictionary to store metrics for each model
metrics_summary = {
    "Model": [],
    "R2": [],
    "MSE": [],
    "MSLE": []  # Mean Absolute Error as an additional metric
}

# Function to load metrics by running the model file
def load_metrics(model_path):
    """Run the model file and capture its output metrics."""
    result = subprocess.run(['python', model_path], capture_output=True, text=True)
    output = result.stdout

    # Extract metrics from the output
    metrics = {}
    for line in output.split('\n'):
        if "R2 score:" in line:
            metrics["R2"] = float(line.split(":")[1].strip())
        elif "MSE:" in line:
            metrics["MSE"] = float(line.split(":")[1].strip())
        elif "MSLE:" in line:
            metrics["MSLE"] = float(line.split(":")[1].strip())
    return metrics

# List of files to exclude from processing
exclude_files = ["accuracies.txt", "compare_models.py", "feature_scaling.py", "__pycache__"]

# Iterate over all files in the models directory
for file in os.listdir(models_dir):
    if file in exclude_files or not file.endswith(".py"):
        continue

    file_path = os.path.join(models_dir, file)

    if os.path.isfile(file_path):  # Ensure it's a file
        # Extract model name from the filename
        model_name = os.path.splitext(file)[0]

        # Load metrics for the model
        metrics = load_metrics(file_path)

        # Append metrics to the summary dictionary
        metrics_summary["Model"].append(model_name)
        metrics_summary["R2"].append(metrics.get("R2", np.nan))
        metrics_summary["MSE"].append(metrics.get("MSE", np.nan))
        metrics_summary["MSLE"].append(metrics.get("MSLE", np.nan))


# Convert the metrics summary to a DataFrame
metrics_df = pd.DataFrame(metrics_summary)

# Plot comparison of metrics
metrics = ["R2", "MSE", "MSLE"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 20))
fig.suptitle("Model Comparison Metrics", fontsize=16)

# Plot each metric
for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.bar(metrics_df["Model"], metrics_df[metric], color='skyblue', edgecolor='black')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric, fontsize=12, labelpad=15, loc='center')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Write results to output.txt
output_file = os.path.join(models_dir, "output.txt")
with open(output_file, "w") as file:
    file.write(f"{'Model':30}{'R2':<10}{'MSE':<10}{'MSLE':<10}\n")
    file.write(f"{'-'*60}\n")
    for i in range(len(metrics_df)):
        file.write(f"{metrics_df['Model'][i]:30}{metrics_df['R2'][i]:<10.4f}{metrics_df['MSE'][i]:<10.4f}{metrics_df['MSLE'][i]:<10.4f}\n")