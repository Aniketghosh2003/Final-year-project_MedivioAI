import numpy as np
import matplotlib.pyplot as plt
import os

def diagnose_dataset(data_dir='../data/xray_validator_data'):
    """Comprehensive dataset diagnosis"""
    
    print("\n" + "="*70)
    print("DATASET DIAGNOSIS")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        print("   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Check shapes
    print("\n2. Checking shapes...")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_val: {y_val.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # Check data types and ranges
    print("\n3. Checking data types and value ranges...")
    print(f"   X_train dtype: {X_train.dtype}")
    print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   y_train dtype: {y_train.dtype}")
    
    if X_train.min() < 0 or X_train.max() > 1.1:
        print("   ⚠️  WARNING: Images should be normalized to [0, 1] range!")
    else:
        print("   ✓ Images are properly normalized")
    
    # Check class distribution
    print("\n4. Checking class distribution...")
    
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    train_class_0 = np.sum(y_train_classes == 0)
    train_class_1 = np.sum(y_train_classes == 1)
    val_class_0 = np.sum(y_val_classes == 0)
    val_class_1 = np.sum(y_val_classes == 1)
    test_class_0 = np.sum(y_test_classes == 0)
    test_class_1 = np.sum(y_test_classes == 1)
    
    print(f"\n   Training Set:")
    print(f"      Class 0 (Non-X-ray): {train_class_0} ({train_class_0/len(y_train)*100:.1f}%)")
    print(f"      Class 1 (X-ray): {train_class_1} ({train_class_1/len(y_train)*100:.1f}%)")
    
    print(f"\n   Validation Set:")
    print(f"      Class 0 (Non-X-ray): {val_class_0} ({val_class_0/len(y_val)*100:.1f}%)")
    print(f"      Class 1 (X-ray): {val_class_1} ({val_class_1/len(y_val)*100:.1f}%)")
    
    print(f"\n   Test Set:")
    print(f"      Class 0 (Non-X-ray): {test_class_0} ({test_class_0/len(y_test)*100:.1f}%)")
    print(f"      Class 1 (X-ray): {test_class_1} ({test_class_1/len(y_test)*100:.1f}%)")
    
    # Calculate imbalance ratio
    ratio_train = max(train_class_0, train_class_1) / min(train_class_0, train_class_1)
    
    print(f"\n   Imbalance Ratio (Train): {ratio_train:.2f}:1")
    
    if ratio_train > 3:
        print("   ⚠️  SEVERE IMBALANCE DETECTED!")
        print("      This is likely causing your model to predict only one class")
        print("      Solution: Use class weights during training")
    elif ratio_train > 1.5:
        print("   ⚠️  Moderate imbalance detected")
        print("      Recommendation: Use class weights")
    else:
        print("   ✓ Classes are reasonably balanced")
    
    # Check for data issues
    print("\n5. Checking for data quality issues...")
    
    # Check for NaN or Inf
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("   ✗ Found NaN or Inf values in training data!")
    else:
        print("   ✓ No NaN or Inf values found")
    
    # Check for identical images
    if len(np.unique(X_train.reshape(len(X_train), -1), axis=0)) < len(X_train) * 0.9:
        print("   ⚠️  Many duplicate images detected!")
    else:
        print("   ✓ No excessive duplicates found")
    
    # Check label format
    print("\n6. Checking label format...")
    if y_train.shape[1] == 2:
        print("   ✓ Labels are in one-hot format (correct)")
    else:
        print("   ✗ Labels are not in correct format!")
    
    # Visualize class distribution
    print("\n7. Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Class distribution
    datasets = ['Train', 'Val', 'Test']
    class_0_counts = [train_class_0, val_class_0, test_class_0]
    class_1_counts = [train_class_1, val_class_1, test_class_1]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    axes[0].bar(x - width/2, class_0_counts, width, label='Non-X-ray', color='skyblue')
    axes[0].bar(x + width/2, class_1_counts, width, label='X-ray', color='coral')
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sample images
    axes[1].axis('off')
    
    # Create subplot grid for sample images
    gs = axes[1].get_gridspec()
    axes[1].remove()
    subfigs = fig.add_subfigure(gs[0, 1])
    
    sample_axes = subfigs.subplots(2, 4)
    
    # Show 4 samples from each class
    class_0_indices = np.where(y_train_classes == 0)[0][:4]
    class_1_indices = np.where(y_train_classes == 1)[0][:4]
    
    for i in range(4):
        if i < len(class_0_indices):
            sample_axes[0, i].imshow(X_train[class_0_indices[i]])
            sample_axes[0, i].set_title('Non-X-ray', fontsize=9)
            sample_axes[0, i].axis('off')
        
        if i < len(class_1_indices):
            sample_axes[1, i].imshow(X_train[class_1_indices[i]])
            sample_axes[1, i].set_title('X-ray', fontsize=9)
            sample_axes[1, i].axis('off')
    
    subfigs.suptitle('Sample Images', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_diagnosis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Visualization saved as 'data_diagnosis.png'")
    plt.show()
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues = []
    
    if ratio_train > 3:
        issues.append("SEVERE class imbalance")
    elif ratio_train > 1.5:
        issues.append("Moderate class imbalance")
    
    if X_train.min() < 0 or X_train.max() > 1.1:
        issues.append("Images not properly normalized")
    
    if train_class_0 < 50 or train_class_1 < 50:
        issues.append("Very small dataset (< 50 samples per class)")
    
    if len(issues) > 0:
        print("\n⚠️  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n📋 RECOMMENDATIONS:")
        if ratio_train > 1.5:
            print("   • Use class_weight in model.fit() to balance training")
            print("   • Use stratified sampling")
        
        if X_train.min() < 0 or X_train.max() > 1.1:
            print("   • Re-run data_processing.py to normalize images properly")
        
        if train_class_0 < 50 or train_class_1 < 50:
            print("   • Collect more data if possible")
            print("   • Use aggressive data augmentation")
            print("   • Consider using a pre-trained model")
    else:
        print("\n✓ No major issues detected!")
        print("   Your data looks good for training")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    diagnose_dataset('../data/xray_validator_data')