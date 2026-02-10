import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import cv2
from data_processing import XRayDataProcessor

# Set this to False to avoid blocking GUI windows when running from terminal
SHOW_PLOTS = False

class XRayEvaluator:
    def __init__(self, model_path):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model file
        """
        self.model = load_model(model_path)
        self.class_names = ['Non-X-ray', 'X-ray']
        print(f"Model loaded from {model_path}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\n" + "="*50)
        print("MODEL EVALUATION ON TEST SET")
        print("="*50)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # Handle different numbers of metrics safely
        test_loss = results[0]
        test_acc = results[1] if len(results) > 1 else None
        test_precision = results[2] if len(results) > 2 else None
        test_recall = results[3] if len(results) > 3 else None
        test_auc = results[4] if len(results) > 4 else None

        print(f"\nTest Loss: {test_loss:.4f}")
        if test_acc is not None:
            print(f"Test Accuracy: {test_acc:.4f}")
        if test_precision is not None:
            print(f"Test Precision: {test_precision:.4f}")
        if test_recall is not None:
            print(f"Test Recall: {test_recall:.4f}")
        if test_auc is not None:
            print(f"Test AUC: {test_auc:.4f}")
        
        # F1 Score
        if test_precision is not None and test_recall is not None and (test_precision + test_recall) > 0:
            f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
            print(f"Test F1-Score: {f1:.4f}")
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"\nConfusion matrix saved to {save_path}")
        
        # Print detailed metrics
        print("\n" + "="*50)
        print("CONFUSION MATRIX BREAKDOWN")
        print("="*50)
        print(f"True Negatives (Non-X-ray correctly classified): {cm[0, 0]}")
        print(f"False Positives (Non-X-ray misclassified as X-ray): {cm[0, 1]}")
        print(f"False Negatives (X-ray misclassified as Non-X-ray): {cm[1, 0]}")
        print(f"True Positives (X-ray correctly classified): {cm[1, 1]}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """Plot ROC curve"""
        # Calculate ROC curve for X-ray class (class 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"\nROC curve saved to {save_path}")
        print(f"AUC Score: {roc_auc:.4f}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                    save_path='precision_recall_curve.png'):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1]
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"\nPrecision-Recall curve saved to {save_path}")
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names,
                                   digits=4))
    
    def visualize_predictions(self, X_test, y_test, num_samples=16, 
                             save_path='predictions_sample.png'):
        """Visualize sample predictions"""
        y_pred_proba = self.model.predict(X_test[:num_samples], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test[:num_samples], axis=1)
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(num_samples):
            ax = axes[i]
            img = X_test[i]
            
            true_label = self.class_names[y_true[i]]
            pred_label = self.class_names[y_pred[i]]
            confidence = y_pred_proba[i][y_pred[i]] * 100
            
            ax.imshow(img)
            ax.axis('off')
            
            # Color code: green for correct, red for incorrect
            color = 'green' if y_true[i] == y_pred[i] else 'red'
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"\nPrediction samples saved to {save_path}")
    
    def predict_single_image(self, image_path, img_size=(224, 224)):
        """Predict class for a single image"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        pred_proba = self.model.predict(img, verbose=0)
        pred_class = np.argmax(pred_proba)
        confidence = pred_proba[0][pred_class] * 100
        
        print(f"\nPrediction for {image_path}:")
        print(f"Class: {self.class_names[pred_class]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities: Non-X-ray={pred_proba[0][0]*100:.2f}%, "
              f"X-ray={pred_proba[0][1]*100:.2f}%")
        
        return pred_class, confidence
    
    def analyze_misclassifications(self, X_test, y_test, save_path='misclassifications.png'):
        """Analyze and visualize misclassified samples"""
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Find misclassified samples
        misclassified_idx = np.where(y_pred != y_true)[0]
        
        if len(misclassified_idx) == 0:
            print("\nNo misclassifications found!")
            return
        
        print(f"\nTotal misclassifications: {len(misclassified_idx)}")
        
        # Show up to 12 misclassified samples
        num_show = min(12, len(misclassified_idx))
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(num_show):
            idx = misclassified_idx[i]
            ax = axes[i]
            img = X_test[idx]
            
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            confidence = y_pred_proba[idx][y_pred[idx]] * 100
            
            ax.imshow(img)
            ax.axis('off')
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        
        # Hide unused subplots
        for i in range(num_show, 12):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()
        
        print(f"\nMisclassifications saved to {save_path}")


def run_complete_evaluation(model_path, test_data_path='../data/xray_validator_data'):
    """Run complete evaluation pipeline"""
    print("\n" + "="*60)
    print("STARTING COMPLETE EVALUATION PIPELINE")
    print("="*60)
    
    # Load test data
    _, _, X_test, _, _, y_test = XRayDataProcessor.load_data(test_data_path)
    
    # Initialize evaluator
    evaluator = XRayEvaluator(model_path)
    
    # Evaluate model
    y_true, y_pred, y_pred_proba = evaluator.evaluate_model(X_test, y_test)
    
    # Generate all visualizations
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_roc_curve(y_true, y_pred_proba)
    evaluator.plot_precision_recall_curve(y_true, y_pred_proba)
    evaluator.print_classification_report(y_true, y_pred)
    evaluator.visualize_predictions(X_test, y_test, num_samples=16)
    evaluator.analyze_misclassifications(X_test, y_test)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Run complete evaluation
    MODEL_PATH = '../models/xray_validator_model.h5'
    
    run_complete_evaluation(MODEL_PATH, '../data/xray_validator_data')
    
    # Optional: Test on single image
    # evaluator = XRayEvaluator(MODEL_PATH)
    # evaluator.predict_single_image('path/to/test/image.jpg')