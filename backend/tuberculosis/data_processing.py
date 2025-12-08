"""
Tuberculosis Dataset Processing Script
Place this in: backend/tuberculosis/data_processing.py
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# Set random seed
random.seed(42)
np.random.seed(42)

# ========== CONFIGURATION ==========
class Config:
    # Source directory (your Kaggle TB dataset)
    SOURCE_DIR = '../data/tuberculosis_raw'  # Update this to your Kaggle dataset path
    
    # Output directory
    OUTPUT_DIR = '../data/tuberculosis'
    
    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Image extensions
    ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

config = Config()

# ========== HELPER FUNCTIONS ==========
def is_image_file(filename: str) -> bool:
    """Check if file is a valid image"""
    name = os.path.basename(filename)
    return (
        not name.startswith(".")
        and os.path.splitext(name)[1].lower() in config.ALLOWED_EXTS
    )

def is_valid_image(path: str) -> bool:
    """Verify if image can be opened"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False

def create_directory_structure(base_dir):
    """Create train/val/test directory structure"""
    splits = ['train', 'val', 'test']
    classes = ['Normal', 'Tuberculosis']
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
    
    print(f"✓ Created directory structure in: {base_dir}")

def scan_dataset(source_dir):
    """
    Scan source directory for Normal and TB images
    Handles multiple dataset structures:
    1. Organized: Normal/ and Tuberculosis/ folders
    2. Mixed: All images with keywords in filenames
    3. Subfolder structure
    """
    
    normal_images = []
    tb_images = []
    
    print(f"\nScanning dataset: {source_dir}")
    
    # Check for organized structure (Normal and Tuberculosis folders)
    normal_folder = os.path.join(source_dir, 'Normal')
    tb_folder = os.path.join(source_dir, 'Tuberculosis')
    
    if os.path.exists(normal_folder) and os.path.exists(tb_folder):
        print("✓ Found organized structure (Normal/ and Tuberculosis/ folders)")
        
        # Scan Normal folder
        for root, dirs, files in os.walk(normal_folder):
            for file in files:
                if is_image_file(file):
                    filepath = os.path.join(root, file)
                    if is_valid_image(filepath):
                        normal_images.append(filepath)
        
        # Scan TB folder
        for root, dirs, files in os.walk(tb_folder):
            for file in files:
                if is_image_file(file):
                    filepath = os.path.join(root, file)
                    if is_valid_image(filepath):
                        tb_images.append(filepath)
    
    else:
        print("⚠ No organized folders found. Scanning all images...")
        
        # Scan all images and classify by filename
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if is_image_file(file):
                    filepath = os.path.join(root, file)
                    
                    if not is_valid_image(filepath):
                        continue
                    
                    filename_lower = file.lower()
                    
                    # Classify based on keywords in filename
                    if any(kw in filename_lower for kw in ['tb', 'tuberculosis', 'positive', 'sick']):
                        tb_images.append(filepath)
                    elif any(kw in filename_lower for kw in ['normal', 'healthy', 'negative']):
                        normal_images.append(filepath)
                    else:
                        # Check parent folder name
                        parent = os.path.basename(os.path.dirname(filepath)).lower()
                        if any(kw in parent for kw in ['tb', 'tuberculosis', 'positive']):
                            tb_images.append(filepath)
                        elif any(kw in parent for kw in ['normal', 'healthy', 'negative']):
                            normal_images.append(filepath)
    
    return normal_images, tb_images

def split_and_copy_data(normal_images, tb_images, output_dir):
    """Split data and copy to train/val/test directories"""
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total Normal images: {len(normal_images)}")
    print(f"Total TB images: {len(tb_images)}")
    print(f"Total images: {len(normal_images) + len(tb_images)}")
    
    if len(normal_images) == 0 or len(tb_images) == 0:
        raise ValueError("❌ Error: One or both classes have no images!")
    
    # Calculate split sizes
    print(f"\nSplit ratios: Train={config.TRAIN_RATIO}, Val={config.VAL_RATIO}, Test={config.TEST_RATIO}")
    
    # Split normal images
    normal_train, normal_temp = train_test_split(
        normal_images,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=42
    )
    normal_val, normal_test = train_test_split(
        normal_temp,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=42
    )
    
    # Split TB images
    tb_train, tb_temp = train_test_split(
        tb_images,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=42
    )
    tb_val, tb_test = train_test_split(
        tb_temp,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=42
    )
    
    # Print split statistics
    print(f"\n{'='*60}")
    print("SPLIT DISTRIBUTION")
    print(f"{'='*60}")
    print(f"TRAIN   - Normal: {len(normal_train):4d} | TB: {len(tb_train):4d} | Total: {len(normal_train) + len(tb_train):4d}")
    print(f"VAL     - Normal: {len(normal_val):4d} | TB: {len(tb_val):4d} | Total: {len(normal_val) + len(tb_val):4d}")
    print(f"TEST    - Normal: {len(normal_test):4d} | TB: {len(tb_test):4d} | Total: {len(normal_test) + len(tb_test):4d}")
    
    # Copy files
    splits_data = {
        'train': {'Normal': normal_train, 'Tuberculosis': tb_train},
        'val': {'Normal': normal_val, 'Tuberculosis': tb_val},
        'test': {'Normal': normal_test, 'Tuberculosis': tb_test}
    }
    
    print(f"\n{'='*60}")
    print("COPYING FILES")
    print(f"{'='*60}")
    
    for split, classes in splits_data.items():
        for class_name, images in classes.items():
            dest_dir = os.path.join(output_dir, split, class_name)
            
            for i, img_path in enumerate(images):
                ext = os.path.splitext(img_path)[1]
                new_filename = f"{class_name}_{split}_{i:04d}{ext}"
                dest_path = os.path.join(dest_dir, new_filename)
                
                shutil.copy2(img_path, dest_path)
            
            print(f"✓ Copied {len(images):4d} images to {split}/{class_name}")

def verify_dataset(output_dir):
    """Verify the created dataset"""
    
    print(f"\n{'='*60}")
    print("DATASET VERIFICATION")
    print(f"{'='*60}")
    
    splits = ['train', 'val', 'test']
    classes = ['Normal', 'Tuberculosis']
    
    for split in splits:
        print(f"\n{split.upper()}:")
        for cls in classes:
            path = os.path.join(output_dir, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if is_image_file(f)])
                print(f"  {cls:15s}: {count:4d} images")
            else:
                print(f"  {cls:15s}: ❌ Directory not found")

def visualize_samples(output_dir, num_samples=5):
    """Visualize sample images from train set"""
    
    print(f"\n{'='*60}")
    print("GENERATING SAMPLE VISUALIZATION")
    print(f"{'='*60}")
    
    train_dir = os.path.join(output_dir, 'train')
    classes = ['Normal', 'Tuberculosis']
    
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 6))
    
    for i, cls in enumerate(classes):
        class_path = os.path.join(train_dir, cls)
        images = [f for f in os.listdir(class_path) if is_image_file(f)][:num_samples]
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('L')
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(cls, fontsize=14, fontweight='bold')
            except Exception as e:
                axes[i, j].text(0.5, 0.5, 'Error loading', ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save in tuberculosis folder
    os.makedirs('../evaluation_results/tuberculosis', exist_ok=True)
    save_path = '../evaluation_results/tuberculosis/sample_images.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample images saved: {save_path}")
    plt.close()

def main():
    """Main execution function"""
    
    print("="*60)
    print("TUBERCULOSIS DATASET PROCESSING")
    print("="*60)
    
    # Check source directory
    if not os.path.exists(config.SOURCE_DIR):
        print(f"\n❌ Error: Source directory not found!")
        print(f"Expected: {config.SOURCE_DIR}")
        print(f"\nPlease:")
        print(f"1. Download TB dataset from Kaggle")
        print(f"2. Extract to: {config.SOURCE_DIR}")
        print(f"3. Update SOURCE_DIR in this script if needed")
        return
    
    # Create output directory structure
    print(f"\nStep 1: Creating directory structure...")
    create_directory_structure(config.OUTPUT_DIR)
    
    # Scan for images
    print(f"\nStep 2: Scanning for images...")
    normal_images, tb_images = scan_dataset(config.SOURCE_DIR)
    
    if len(normal_images) == 0 or len(tb_images) == 0:
        print(f"\n❌ Error: Could not find images for both classes")
        print(f"Normal images found: {len(normal_images)}")
        print(f"TB images found: {len(tb_images)}")
        print(f"\nPlease check:")
        print(f"1. Source directory: {config.SOURCE_DIR}")
        print(f"2. Dataset has 'Normal' and 'Tuberculosis' folders")
        print(f"3. Or images have keywords (tb, tuberculosis, normal) in filenames")
        return
    
    # Split and copy data
    print(f"\nStep 3: Splitting and copying data...")
    split_and_copy_data(normal_images, tb_images, config.OUTPUT_DIR)
    
    # Verify dataset
    print(f"\nStep 4: Verifying dataset...")
    verify_dataset(config.OUTPUT_DIR)
    
    # Visualize samples
    print(f"\nStep 5: Visualizing samples...")
    visualize_samples(config.OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("✅ DATASET PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nProcessed data location: {config.OUTPUT_DIR}")
    print(f"\nNext step: Run train_model.py to train the TB detection model")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()