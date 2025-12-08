import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Pneumonia')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'pneumonia')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32

def load_test_data():
    """Load test data"""
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(LOGS_DIR, exist_ok=True)
    cm_path = os.path.join(LOGS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved as '{cm_path}'")
    plt.close()

def main():
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    print("\nLoading best model...")
    model_path = os.path.join(MODELS_DIR, 'pneumonia_model_best.keras')
    model = keras.models.load_model(model_path)
    
    # Load test data
    print("\nLoading test data...")
    test_gen = load_test_data()
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    
    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("\n" + "=" * 50)
    print("DETAILED METRICS")
    print("=" * 50)
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()