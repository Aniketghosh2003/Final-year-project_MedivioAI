"""
Tuberculosis Detection Model Evaluation Script
Place this in: backend/tuberculosis/evaluate_model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, matthews_corrcoef
)
import os
import json
from datetime import datetime

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# ========== CONFIGURATION ==========
class Config:
    # Paths
    DATA_DIR = '../data/tuberculosis'
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_PATH = '../models/tuberculosis/tb_detection_model_final.h5'
    
    # Evaluation parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    
    # Output directory
    RESULTS_DIR = '../evaluation_results/tuberculosis'

config = Config()

# Create results directory
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ========== LOAD MODEL ==========
def load_model():
    """Load trained model"""
    
    print("Loading trained model...")
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ Error: Model not found at {config.MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return None
    
    model = keras.models.load_model(config.MODEL_PATH)
    print(f"✓ Model loaded: {config.MODEL_PATH}")
    
    return model

def create_test_generator():
    """Create test data generator"""
    
    print("\nCreating test data generator...")
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Class indices: {test_generator.class_indices}")
    
    return test_generator

# ========== EVALUATION ==========
def evaluate_model(model, test_generator):
    """Evaluate model and get predictions"""
    
    print(f"\n{'='*60}")
    print("EVALUATING MODEL")
    print(f"{'='*60}")
    
    print("\nGenerating predictions...")
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    print("\nCalculating metrics...")
    results = model.evaluate(test_generator, verbose=1)
    
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4],
        'f1_score': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    return y_true, y_pred, y_pred_prob, metrics

# ========== CONFUSION MATRIX ==========
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Tuberculosis'],
        yticklabels=['Normal', 'Tuberculosis'],
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 16}
    )
    
    plt.title('Confusion Matrix - TB Detection', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(
        1, -0.3,
        f'Overall Accuracy: {accuracy:.2%}',
        ha='center',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    return cm

# ========== ROC CURVE ==========
def plot_roc_curve(y_true, y_pred_prob):
    """Plot ROC curve"""
    
    print("\nGenerating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curve - TB Detection', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    roc_path = os.path.join(config.RESULTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: {roc_path}")
    plt.close()
    
    return roc_auc

# ========== PR CURVE ==========
def plot_precision_recall_curve(y_true, y_pred_prob):
    """Plot Precision-Recall curve"""
    
    print("\nGenerating Precision-Recall curve...")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plt.plot(
        recall, precision,
        color='blue',
        lw=2,
        label=f'PR curve (AUC = {pr_auc:.3f})'
    )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curve - TB Detection', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    pr_path = os.path.join(config.RESULTS_DIR, 'precision_recall_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall curve saved: {pr_path}")
    plt.close()
    
    return pr_auc

# ========== CLASSIFICATION REPORT ==========
def generate_classification_report(y_true, y_pred):
    """Generate classification report"""
    
    print("\nGenerating classification report...")
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Normal', 'Tuberculosis'],
        digits=4
    )
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(report)
    
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("TUBERCULOSIS DETECTION - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"\n✓ Classification report saved: {report_path}")
    
    return report

# ========== METRICS VISUALIZATION ==========
def plot_metrics_comparison(metrics):
    """Plot all metrics"""
    
    print("\nGenerating metrics comparison plot...")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'MCC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc'],
        (metrics['mcc'] + 1) / 2  # Normalize MCC
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Model Performance Metrics - TB Detection', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    metrics_path = os.path.join(config.RESULTS_DIR, 'metrics_comparison.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved: {metrics_path}")
    plt.close()

# ========== SAVE RESULTS ==========
def save_evaluation_results(metrics, cm, roc_auc, pr_auc):
    """Save evaluation results to JSON"""
    
    print("\nSaving evaluation results...")
    
    results = {
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': config.MODEL_PATH,
        'test_samples': int(cm.sum()),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc_roc': float(metrics['auc']),
            'auc_pr': float(pr_auc),
            'mcc': float(metrics['mcc']),
            'loss': float(metrics['loss'])
        },
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        },
        'class_performance': {
            'normal': {
                'samples': int(cm[0].sum()),
                'correctly_classified': int(cm[0, 0]),
                'misclassified': int(cm[0, 1])
            },
            'tuberculosis': {
                'samples': int(cm[1].sum()),
                'correctly_classified': int(cm[1, 1]),
                'misclassified': int(cm[1, 0])
            }
        }
    }
    
    results_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Evaluation results saved: {results_path}")

# ========== SUMMARY ==========
def print_summary(metrics, cm):
    """Print evaluation summary"""
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:   {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"   AUC-ROC:    {metrics['auc']:.4f}")
    print(f"   MCC:        {metrics['mcc']:.4f}")
    print(f"   Loss:       {metrics['loss']:.4f}")
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"   True Negatives:   {cm[0, 0]}")
    print(f"   False Positives:  {cm[0, 1]}")
    print(f"   False Negatives:  {cm[1, 0]}")
    print(f"   True Positives:   {cm[1, 1]}")
    
    print(f"\n📁 Results saved in: {config.RESULTS_DIR}")
    
    print(f"\n{'='*60}")

# ========== MAIN ==========
def main():
    """Main evaluation pipeline"""
    
    print("="*60)
    print("TUBERCULOSIS DETECTION MODEL EVALUATION")
    print("="*60)
    
    if not os.path.exists(config.TEST_DIR):
        print(f"\n❌ Error: Test directory not found: {config.TEST_DIR}")
        print("Please run data_processing.py first!")
        return
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create test generator
    test_generator = create_test_generator()
    
    # Evaluate model
    y_true, y_pred, y_pred_prob, metrics = evaluate_model(model, test_generator)
    
    # Generate visualizations
    cm = plot_confusion_matrix(y_true, y_pred)
    roc_auc = plot_roc_curve(y_true, y_pred_prob)
    pr_auc = plot_precision_recall_curve(y_true, y_pred_prob)
    
    # Generate classification report
    generate_classification_report(y_true, y_pred)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics)
    
    # Save results
    save_evaluation_results(metrics, cm, roc_auc, pr_auc)
    
    # Print summary
    print_summary(metrics, cm)
    
    print("\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"\n📊 All results saved in: {config.RESULTS_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()