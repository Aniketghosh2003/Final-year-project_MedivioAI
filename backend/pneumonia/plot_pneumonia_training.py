import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Build path to your pneumonia training log relative to this file
BASE_DIR = Path(__file__).resolve().parents[1]  # points to backend/
log_path = BASE_DIR / "logs" / "pneumonia" / "training_log_20251107_134851.csv"

# Read CSV
df = pd.read_csv(log_path)

# Drop completely empty rows (in case there are blanks)
df = df.dropna(subset=["epoch"])

# Sort by epoch just in case
df = df.sort_values("epoch")

epochs = df["epoch"]

# 1) Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["accuracy"], label="Train Accuracy")
plt.plot(epochs, df["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Pneumonia Model Accuracy")
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
plt.title("Pneumonia Model Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Precision & Recall (optional)
plt.figure(figsize=(8, 5))
plt.plot(epochs, df["precision"], label="Train Precision")
plt.plot(epochs, df["recall"], label="Train Recall")
plt.plot(epochs, df["val_precision"], label="Val Precision")
plt.plot(epochs, df["val_recall"], label="Val Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Pneumonia Model Precision & Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()