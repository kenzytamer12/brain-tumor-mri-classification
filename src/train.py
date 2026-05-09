import os
import json
import time
import argparse
from datetime import datetime

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.data.preprocessing import get_generators
from src.models.cnn_from_scratch import build_cnn_from_scratch
from src.models.transfer_learning import build_transfer_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def plot_history(history, model_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_history.png"))
    plt.close()


def count_trainable_params(model):
    return int(sum([w.shape.num_elements() for w in model.trainable_weights]))


def train_model(model_name, model, train_gen, val_gen, epochs=30):
    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    best_model_path = os.path.join(MODEL_DIR, f"best_{model_name}.h5")
    final_model_path = os.path.join(MODEL_DIR, f"final_{model_name}.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(
                LOGS_DIR,
                f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        )
    ]

    start_time = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    model.save(final_model_path)
    plot_history(history, model_name)

    metrics = {
        "model_name": model_name,
        "best_train_accuracy": float(max(history.history["accuracy"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "training_time_seconds": round(training_time, 2),
        "trainable_parameters": count_trainable_params(model)
    }

    with open(os.path.join(RESULTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(RESULTS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history.history, f, indent=4)

    print(f"\n✅ {model_name} training completed!")
    print(f"Best Validation Accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"Training Time: {metrics['training_time_seconds']} seconds")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "both"],
        default="cnn",
        help="Choose which model to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    all_metrics = []

    if args.model in ["cnn", "both"]:
        print("\nLoading generators for CNN from scratch...")
        train_gen_cnn, val_gen_cnn, test_gen_cnn = get_generators(use_resnet=False)

        cnn_model = build_cnn_from_scratch(num_classes=train_gen_cnn.num_classes)

        cnn_metrics = train_model(
            "CNN_From_Scratch",
            cnn_model,
            train_gen_cnn,
            val_gen_cnn,
            epochs=args.epochs
        )

        all_metrics.append(cnn_metrics)

    if args.model in ["resnet", "both"]:
        print("\nLoading generators for ResNet50 transfer learning...")
        train_gen_resnet, val_gen_resnet, test_gen_resnet = get_generators(use_resnet=True)

        resnet_model = build_transfer_model(num_classes=train_gen_resnet.num_classes)

        resnet_metrics = train_model(
            "ResNet50_Transfer",
            resnet_model,
            train_gen_resnet,
            val_gen_resnet,
            epochs=args.epochs
        )

        all_metrics.append(resnet_metrics)

    with open(os.path.join(RESULTS_DIR, "all_training_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for m in all_metrics:
        print(f"{m['model_name']}:")
        print(f"  Best Train Accuracy: {m['best_train_accuracy']:.4f}")
        print(f"  Best Validation Accuracy: {m['best_val_accuracy']:.4f}")
        print(f"  Training Time: {m['training_time_seconds']} seconds")
        print(f"  Trainable Parameters: {m['trainable_parameters']}")
        print()


if __name__ == "__main__":
    main()