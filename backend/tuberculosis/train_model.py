"""
Tuberculosis Detection Model Training Script
Place this in: backend/tuberculosis/train_model.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ========== CONFIGURATION ==========
class Config:
    # Paths
    DATA_DIR = '../data/tuberculosis'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    
    # Model save paths
    MODEL_SAVE_DIR = '../models/tuberculosis'
    LOGS_DIR = '../logs/tuberculosis'
    MODEL_NAME = 'tb_detection_model'
    
    # Training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Model architecture
    BASE_MODEL = 'DenseNet121'
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 256
    
    # Data augmentation
    ROTATION_RANGE = 20
    WIDTH_SHIFT = 0.2
    HEIGHT_SHIFT = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True

config = Config()

# Create directories
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)

# ========== DATA GENERATORS ==========
def create_data_generators():
    """Create data generators"""
    
    print("Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT,
        height_shift_range=config.HEIGHT_SHIFT,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        shear_range=0.15,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator

# ========== MODEL ==========
def create_model():
    """Create TB detection model"""
    
    print(f"\nCreating model with {config.BASE_MODEL}...")
    
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE * 0.6),
        layers.Dense(1, activation='sigmoid')
    ], name='TB_Detection_Model')
    
    print("✓ Model created successfully")
    
    return model

# ========== CALLBACKS ==========
def create_callbacks():
    """Create training callbacks"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_NAME}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'tb_model_{timestamp}'),
            histogram_freq=1
        )
    ]
    
    return callbacks

# ========== TRAINING ==========
def train_model(model, train_gen, val_gen, callbacks):
    """Train the model (Phase 1: Frozen base)"""
    
    print(f"\n{'='*60}")
    print("PHASE 1: TRAINING WITH FROZEN BASE MODEL")
    print(f"{'='*60}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def fine_tune_model(model, train_gen, val_gen, callbacks, initial_history):
    """Fine-tune the model (Phase 2: Unfrozen layers)"""
    
    print(f"\n{'='*60}")
    print("PHASE 2: FINE-TUNING WITH UNFROZEN LAYERS")
    print(f"{'='*60}")
    
    model.layers[0].trainable = True
    
    for layer in model.layers[0].layers[:-50]:
        layer.trainable = False
    
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_count:,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    for key in initial_history.history.keys():
        initial_history.history[key].extend(history_fine.history[key])
    
    return initial_history

# ========== SAVE ==========
def save_model_and_config(model, history):
    """Save model and configuration"""
    
    final_model_path = os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_NAME}_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Model saved: {final_model_path}")
    
    history_path = os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_NAME}_history.json')
    with open(history_path, 'w') as f:
        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        json.dump(history_dict, f, indent=4)
    print(f"✓ Training history saved: {history_path}")
    
    config_dict = {
        'img_size': config.IMG_SIZE,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'base_model': config.BASE_MODEL,
        'dropout_rate': config.DROPOUT_RATE,
        'dense_units': config.DENSE_UNITS,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_path = os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_NAME}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Configuration saved: {config_path}")

# ========== VISUALIZATION ==========
def plot_training_history(history):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    titles = ['Model Accuracy', 'Model Loss', 'Model Precision', 'Model Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].plot(history.history[metric], label='Train', linewidth=2)
        axes[row, col].plot(history.history[f'val_{metric}'], label='Validation', linewidth=2)
        axes[row, col].set_title(title, fontsize=14, fontweight='bold')
        axes[row, col].set_xlabel('Epoch', fontsize=12)
        axes[row, col].set_ylabel(metric.capitalize(), fontsize=12)
        axes[row, col].legend(fontsize=11)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.MODEL_SAVE_DIR, f'{config.MODEL_NAME}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plot saved: {plot_path}")
    plt.close()

# ========== MAIN ==========
def main():
    """Main training pipeline"""
    
    print("="*60)
    print("TUBERCULOSIS DETECTION MODEL TRAINING")
    print("="*60)
    
    if not os.path.exists(config.TRAIN_DIR) or not os.path.exists(config.VAL_DIR):
        print(f"\n❌ Error: Data directories not found!")
        print(f"Expected:")
        print(f"  Train: {config.TRAIN_DIR}")
        print(f"  Val: {config.VAL_DIR}")
        print(f"\nPlease run data_processing.py first!")
        return
    
    # Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Create model
    model = create_model()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model (Phase 1)
    history = train_model(model, train_generator, val_generator, callbacks)
    
    # Fine-tune model (Phase 2)
    history = fine_tune_model(model, train_generator, val_generator, callbacks, history)
    
    # Save model and configuration
    save_model_and_config(model, history)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\nModel saved in: {config.MODEL_SAVE_DIR}")
    print(f"\nNext step: Run evaluate_model.py to evaluate on test data")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()