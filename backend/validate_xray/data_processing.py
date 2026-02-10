import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import json

class XRayDataProcessor:
    def __init__(self, xray_dir, non_xray_dir, img_size=(224, 224)):
        """
        Initialize data processor for X-ray classification
        
        Args:
            xray_dir: Directory containing X-ray images
            non_xray_dir: Directory containing non-X-ray images
            img_size: Target image size (height, width)
        """
        self.xray_dir = xray_dir
        self.non_xray_dir = non_xray_dir
        self.img_size = img_size
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_images(self, directory, label, max_images=None):
        """Load and preprocess images from directory.

        Args:
            directory: folder with images
            label: numeric label to assign
            max_images: optional cap on number of images to load (per class)
        """
        images = []
        labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        print(f"Loading images from {directory}...")
        files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

        # Optionally limit how many images we load to avoid running out of RAM
        if max_images is not None and len(files) > max_images:
            print(f"Found {len(files)} files, sampling {max_images} to limit memory usage...")
            files = random.sample(files, max_images)
        
        for filename in files:
            img_path = os.path.join(directory, filename)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {filename}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, self.img_size)
                
                # Normalize to [0, 1]
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        print(f"Loaded {len(images)} images with label {label}")
        return np.array(images), np.array(labels)
    
    def prepare_data(self, test_size=0.15, val_size=0.15, random_state=42, max_images_per_class=None):
        """Load and split data into train/val/test sets"""
        # Load X-ray images (label = 1)
        xray_images, xray_labels = self.load_images(self.xray_dir, 1, max_images=max_images_per_class)
        
        # Load non-X-ray images (label = 0)
        non_xray_images, non_xray_labels = self.load_images(self.non_xray_dir, 0, max_images=max_images_per_class)
        
        # Combine datasets
        X = np.concatenate([xray_images, non_xray_images], axis=0)
        y = np.concatenate([xray_labels, non_xray_labels], axis=0)
        
        print(f"\nTotal dataset size: {len(X)} images")
        print(f"X-ray images: {np.sum(y == 1)}")
        print(f"Non-X-ray images: {np.sum(y == 0)}")
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        # Convert labels to categorical
        self.y_train = to_categorical(self.y_train, 2)
        self.y_val = to_categorical(self.y_val, 2)
        self.y_test = to_categorical(self.y_test, 2)
        
        print(f"\nTrain set: {len(self.X_train)} images")
        print(f"Validation set: {len(self.X_val)} images")
        print(f"Test set: {len(self.X_test)} images")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_data_augmentation(self):
        """Create data augmentation generator for training"""
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        return datagen
    
    def save_data(self, save_dir='processed_data'):
        """Save processed data to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), self.X_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), self.X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(save_dir, 'y_val.npy'), self.y_val)
        np.save(os.path.join(save_dir, 'y_test.npy'), self.y_test)
        
        # Save metadata
        metadata = {
            'img_size': self.img_size,
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test)
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\nData saved to {save_dir}")
    
    @staticmethod
    def load_data(save_dir='processed_data'):
        """Load processed data from disk"""
        X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(save_dir, 'y_test.npy'))
        
        print("Data loaded successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    XRAY_DIR = "../data/xray_validator_data_raw_data/x-ray"
    NON_XRAY_DIR = "../data/xray_validator_data_raw_data/non-x-ray"
    SAVE_DIR = "../data/xray_validator_data"
    
    # Initialize processor
    processor = XRayDataProcessor(XRAY_DIR, NON_XRAY_DIR, img_size=(224, 224))
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.prepare_data(
        test_size=0.15,
        val_size=0.15,
        max_images_per_class=2250  # limit per class to avoid running out of RAM
    )
    
    # Save processed data
    processor.save_data(SAVE_DIR)
    
    print("\nData processing complete!")