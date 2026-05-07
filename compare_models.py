import os
import json
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


cnn_eval = load_json("results/CNN_From_Scratch_evaluation_metrics.json")
resnet_eval = load_json("results/ResNet50_Transfer_evaluation_metrics.json")

cnn_train = load_json("results/CNN_From_Scratch_metrics.json")
resnet_train = load_json("results/ResNet50_Transfer_metrics.json")

data = [
    {
        "Model": "CNN From Scratch",
        "Accuracy": cnn_eval.get("Accuracy", "N/A"),
        "Precision": cnn_eval.get("Precision", "N/A"),
        "Recall": cnn_eval.get("Recall", "N/A"),
        "F1 Score": cnn_eval.get("F1 Score", "N/A"),
        "Best Train Accuracy": cnn_train.get("best_train_accuracy", "N/A"),
        "Best Validation Accuracy": cnn_train.get("best_val_accuracy", "N/A"),
        "Training Time Seconds": cnn_train.get("training_time_seconds", "N/A"),
        "Trainable Parameters": cnn_train.get("trainable_parameters", "N/A")
    },
    {
        "Model": "ResNet50 Transfer",
        "Accuracy": resnet_eval.get("Accuracy", "N/A"),
        "Precision": resnet_eval.get("Precision", "N/A"),
        "Recall": resnet_eval.get("Recall", "N/A"),
        "F1 Score": resnet_eval.get("F1 Score", "N/A"),
        "Best Train Accuracy": resnet_train.get("best_train_accuracy", "N/A"),
        "Best Validation Accuracy": resnet_train.get("best_val_accuracy", "N/A"),
        "Training Time Seconds": resnet_train.get("training_time_seconds", "N/A"),
        "Trainable Parameters": resnet_train.get("trainable_parameters", "N/A")
    }
]

df = pd.DataFrame(data)

print("\nModel Comparison:\n")
print(df)

csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)

# Chart for evaluation metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

plot_df = df[["Model"] + metrics].copy()

for metric in metrics:
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

plot_df.set_index("Model")[metrics].plot(kind="bar", figsize=(10, 6))
plt.title("Model Evaluation Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()

chart_path = os.path.join(RESULTS_DIR, "model_comparison_chart.png")
plt.savefig(chart_path)
plt.close()

print(f"\n Saved comparison CSV: {csv_path}")
print(f" Saved comparison chart: {chart_path}")