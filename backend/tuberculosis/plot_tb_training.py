import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Build path to backend directory
BASE_DIR = Path(__file__).resolve().parents[1]  # points to backend/

# Path to the saved TB training history JSON
history_path = BASE_DIR / "models" / "tuberculosis" / "tb_detection_model_history.json"

if not history_path.exists():
    raise FileNotFoundError(
        f"Training history file not found at {history_path}.\n"
        "Make sure you have run the TB training script so that the history JSON is saved."
    )

# Load JSON history (keys like 'accuracy', 'val_accuracy', 'loss', 'val_loss', etc.)
with open(history_path, "r") as f:
    history = json.load(f)

# Convert to DataFrame for convenience and add epoch index (1-based)
df = pd.DataFrame(history)
df["epoch"] = range(1, len(df) + 1)

epochs = df["epoch"]

# 1) Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["accuracy"], label="Train Accuracy")
plt.plot(epochs, df["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Tuberculosis Model Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Loss plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["loss"], label="Train Loss")
plt.plot(epochs, df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tuberculosis Model Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Precision & Recall (if available)
if all(col in df.columns for col in ["precision", "recall", "val_precision", "val_recall"]):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["precision"], label="Train Precision")
    plt.plot(epochs, df["recall"], label="Train Recall")
    plt.plot(epochs, df["val_precision"], label="Val Precision")
    plt.plot(epochs, df["val_recall"], label="Val Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Tuberculosis Model Precision & Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Precision/Recall columns not found in history; skipping that plot.")
