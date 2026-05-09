import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data.preprocessing import get_generators


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


MODELS = {
    "CNN_From_Scratch": os.path.join(MODEL_DIR, "best_CNN_From_Scratch.h5"),
    "ResNet50_Transfer": os.path.join(MODEL_DIR, "best_ResNet50_Transfer.h5")
}


def evaluate_model(model_name, model_path, test_gen, class_names):
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name} on TEST SET")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    model = load_model(model_path)

    test_gen.reset()

    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    report_path = os.path.join(RESULTS_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    results = {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1 Score": round(float(f1), 4)
    }

    json_path = os.path.join(RESULTS_DIR, f"{model_name}_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved confusion matrix: {cm_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved metrics: {json_path}")

    return results

print("Starting model evaluation...")

def main():
    all_results = []
    print("Starting model evaluation...")
    # =========================
    # EVALUATE CNN ON TEST SET
    # =========================
    print("\nLoading CNN test generator...")
    _, _, test_gen_cnn = get_generators(use_resnet=False)

    class_names_cnn = list(test_gen_cnn.class_indices.keys())

    print("\nCNN Class mapping:")
    print(test_gen_cnn.class_indices)

    cnn_result = evaluate_model(
        "CNN_From_Scratch",
        MODELS["CNN_From_Scratch"],
        test_gen_cnn,
        class_names_cnn
    )

    if cnn_result is not None:
        all_results.append(cnn_result)

    # =========================
    # EVALUATE RESNET50 ON TEST SET
    # =========================
    print("\nLoading ResNet50 test generator...")
    _, _, test_gen_resnet = get_generators(use_resnet=True)

    class_names_resnet = list(test_gen_resnet.class_indices.keys())

    print("\nResNet50 Class mapping:")
    print(test_gen_resnet.class_indices)

    resnet_result = evaluate_model(
        "ResNet50_Transfer",
        MODELS["ResNet50_Transfer"],
        test_gen_resnet,
        class_names_resnet
    )

    if resnet_result is not None:
        all_results.append(resnet_result)

    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 60)

    for result in all_results:
        print(result)

    print(f"\n✅ Saved final summary: {summary_path}")


if __name__ == "__main__":
    main()