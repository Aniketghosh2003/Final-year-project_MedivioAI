import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# ============================================================================
# CRITICAL FIX 1: Check data balance
# ============================================================================
def check_data_balance(y_train, y_val, y_test):
    """Check if dataset is balanced"""
    print("\n" + "="*60)
    print("DATA BALANCE CHECK")
    print("="*60)
    
    train_xray = np.sum(np.argmax(y_train, axis=1) == 1)
    train_non_xray = np.sum(np.argmax(y_train, axis=1) == 0)
    
    val_xray = np.sum(np.argmax(y_val, axis=1) == 1)
    val_non_xray = np.sum(np.argmax(y_val, axis=1) == 0)
    
    test_xray = np.sum(np.argmax(y_test, axis=1) == 1)
    test_non_xray = np.sum(np.argmax(y_test, axis=1) == 0)
    
    print(f"\nTrain Set:")
    print(f"  X-ray: {train_xray} ({train_xray/len(y_train)*100:.1f}%)")
    print(f"  Non-X-ray: {train_non_xray} ({train_non_xray/len(y_train)*100:.1f}%)")
    
    print(f"\nValidation Set:")
    print(f"  X-ray: {val_xray} ({val_xray/len(y_val)*100:.1f}%)")
    print(f"  Non-X-ray: {val_non_xray} ({val_non_xray/len(y_val)*100:.1f}%)")
    
    print(f"\nTest Set:")
    print(f"  X-ray: {test_xray} ({test_xray/len(y_test)*100:.1f}%)")
    print(f"  Non-X-ray: {test_non_xray} ({test_non_xray/len(y_test)*100:.1f}%)")
    
    # Warning for severe imbalance
    ratio = max(train_xray, train_non_xray) / min(train_xray, train_non_xray)
    if ratio > 2:
        print(f"\n⚠️  WARNING: Dataset is imbalanced (ratio: {ratio:.2f}:1)")
        print("    Solution: Using class weights to balance training")
    
    return train_xray, train_non_xray


# ============================================================================
# CRITICAL FIX 2: Calculate class weights for imbalanced data
# ============================================================================
def get_class_weights(y_train):
    """Calculate class weights to handle imbalance"""
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"\nClass weights: {class_weight_dict}")
    return class_weight_dict


# ============================================================================
# CRITICAL FIX 3: Better model with proper regularization
# ============================================================================
def build_robust_model(img_size=(224, 224)):
    """Build a more robust model"""
    # Use EfficientNetB0 with proper configuration
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = Input(shape=(*img_size, 3))
    
    # Preprocessing: normalize to [-1, 1] for EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer with sigmoid for binary classification
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model


# ============================================================================
# CRITICAL FIX 4: Stronger data augmentation
# ============================================================================
def get_strong_augmentation():
    """Create stronger data augmentation"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )


# ============================================================================
# CRITICAL FIX 5: Better training configuration
# ============================================================================
def train_robust_model(X_train, y_train, X_val, y_val, 
                       epochs=100, batch_size=32):
    """Train with proper configuration"""
    
    print("\n" + "="*60)
    print("BUILDING ROBUST MODEL")
    print("="*60)
    
    # Build model
    model, base_model = build_robust_model()
    
    # Compile with appropriate learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower LR for stability
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("✓ Model compiled")
    model.summary()
    
    # Get class weights
    class_weights = get_class_weights(y_train)
    
    # Setup callbacks
    models_dir = os.path.join('..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'xray_validator_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Data augmentation
    datagen = get_strong_augmentation()
    
    print("\n" + "="*60)
    print("TRAINING WITH CLASS WEIGHTS AND AUGMENTATION")
    print("="*60)
    
    # Train with class weights
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,  # THIS IS CRITICAL!
        verbose=1,
        steps_per_epoch=len(X_train) // batch_size
    )
    
    print("\n✓ Initial training completed")
    
    # Fine-tuning phase
    print("\n" + "="*60)
    print("FINE-TUNING MODEL")
    print("="*60)
    
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("✓ Model recompiled for fine-tuning")
    
    # Fine-tune
    history_fine = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        steps_per_epoch=len(X_train) // batch_size
    )
    
    return model, history, history_fine


# ============================================================================
# CRITICAL FIX 6: Verify predictions after training
# ============================================================================
def verify_predictions(model, X_val, y_val):
    """Verify model is actually learning"""
    print("\n" + "="*60)
    print("PREDICTION VERIFICATION")
    print("="*60)
    
    y_pred_proba = model.predict(X_val[:100], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_val[:100], axis=1)
    
    print(f"\nFirst 100 predictions:")
    print(f"Predicted X-ray: {np.sum(y_pred == 1)}")
    print(f"Predicted Non-X-ray: {np.sum(y_pred == 0)}")
    print(f"Actual X-ray: {np.sum(y_true == 1)}")
    print(f"Actual Non-X-ray: {np.sum(y_true == 0)}")
    
    # Show some probability distributions
    print(f"\nSample probabilities (first 10):")
    for i in range(min(10, len(y_pred_proba))):
        print(f"  Image {i}: Non-X-ray={y_pred_proba[i][0]:.3f}, "
              f"X-ray={y_pred_proba[i][1]:.3f} | "
              f"True={y_true[i]}, Pred={y_pred[i]}")


# ============================================================================
# MAIN TRAINING SCRIPT WITH ALL FIXES
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FIXED X-RAY CLASSIFIER TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data_dir = os.path.join('..', 'data', 'xray_validator_data')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"✓ Data loaded")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Check data balance
    check_data_balance(y_train, y_val, y_test)
    
    # Train model
    model, history, history_fine = train_robust_model(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Verify predictions
    verify_predictions(model, X_val, y_val)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    results = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    print(f"  AUC: {results[4]:.4f}")
    
    # Save model where backend and evaluation expect it
    final_model_path = os.path.join('..', 'models', 'xray_validator_model.h5')
    model.save(final_model_path)
    print(f"\n Model saved: {final_model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext: Run evaluate.py with the new model")