# ================================================================
# Pneumonia Detection from Chest X-Ray Images
# Binary Image Classification using Deep Learning
# Module: COMP 20037 - Artificial Intelligence and Deep Learning
# ================================================================
# Dataset : Chest X-Ray Images (Pneumonia) - Kaggle
# Link    : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Classes : NORMAL (0) vs PNEUMONIA (1)
# ================================================================

# --- Step 1: Import Libraries ---
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score,
                              precision_score, recall_score)
import seaborn as sns

# Fix random seed so results are reproducible
tf.random.set_seed(42)
np.random.seed(42)

# ================================================================
# Step 2: Set Paths and Settings
# ================================================================

# Image settings
IMG_SIZE   = (128, 128)   # Resize every image to 128x128 pixels
BATCH_SIZE = 32           # Number of images processed at once
EPOCHS     = 20           # Maximum training epochs

# Dataset folders (download from Kaggle and place in project folder)
# Expected structure:
#   chest_xray/
#       train/
#           NORMAL/
#           PNEUMONIA/
#       val/
#           NORMAL/
#           PNEUMONIA/
#       test/
#           NORMAL/
#           PNEUMONIA/

TRAIN_DIR = 'chest_xray/train'
VAL_DIR   = 'chest_xray/val'
TEST_DIR  = 'chest_xray/test'

# ================================================================
# Step 3: Load and Prepare the Data
# ================================================================

# Training data: normalize pixels + apply augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # Scale pixel values from 0-255 to 0-1
    horizontal_flip=True,     # Randomly flip images left-right
    rotation_range=10,        # Randomly rotate images up to 10 degrees
    zoom_range=0.1,           # Randomly zoom in slightly
    width_shift_range=0.1,    # Randomly shift image left or right
    height_shift_range=0.1    # Randomly shift image up or down
)

# Validation and test data: only normalize, no augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training images from folder
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,     # Resize all images
    batch_size=BATCH_SIZE,
    class_mode='binary'       # Binary: 0 = NORMAL, 1 = PNEUMONIA
)

# Load validation images
val_data = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test images (used only for final evaluation)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False             # Keep order for correct label matching
)

# Print dataset summary
print("=" * 55)
print("DATASET SUMMARY")
print("=" * 55)
print(f"Classes        : {train_data.class_indices}")
print(f"Training images: {train_data.samples}")
print(f"Validation imgs: {val_data.samples}")
print(f"Test images    : {test_data.samples}")
print("=" * 55)

# ================================================================
# Step 4: Build Custom CNN Model
# ================================================================

def build_custom_cnn():
    """
    A simple 3-block CNN for binary image classification.
    Each block: Conv2D -> MaxPooling -> Dropout
    Final layers: Flatten -> Dense -> Output
    """
    model = Sequential([

        # Block 1 - Detect basic features like edges
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2 - Detect medium-level patterns
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3 - Detect complex patterns (lung opacity, etc.)
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Flatten and classify
        Flatten(),
        Dense(128, activation='relu'),  # Fully connected layer
        Dropout(0.5),                   # Prevent overfitting
        Dense(1, activation='sigmoid')  # Output: 0 = Normal, 1 = Pneumonia
    ])

    # Compile with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create and display the model
custom_model = build_custom_cnn()
print("\nCUSTOM CNN SUMMARY:")
custom_model.summary()

# ================================================================
# Step 5: Train Custom CNN
# ================================================================

# Stop training early if validation loss does not improve
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,                  # Stop after 4 epochs with no improvement
    restore_best_weights=True    # Keep the best version of the model
)

print("\n" + "=" * 55)
print("TRAINING CUSTOM CNN...")
print("=" * 55)

history_cnn = custom_model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop]
)

# ================================================================
# Step 6: Build MobileNetV2 Transfer Learning Model
# ================================================================

def build_mobilenet_model():
    """
    Uses MobileNetV2 pre-trained on ImageNet as a feature extractor.
    All base layers are frozen. Only the new top layers are trained.
    """
    # Load MobileNetV2 without the top classification layer
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,       # Remove the original classification head
        weights='imagenet'       # Use weights learned from ImageNet
    )
    base.trainable = False       # Freeze all base layers

    # Add a new classification head
    model = Sequential([
        base,
        GlobalAveragePooling2D(),        # Reduce feature map to a single vector
        Dense(128, activation='relu'),   # Fully connected layer
        Dropout(0.4),                    # Regularization
        Dense(1, activation='sigmoid')   # Binary output
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

mobilenet_model = build_mobilenet_model()
print("\nMOBILENETv2 TRANSFER LEARNING SUMMARY:")
mobilenet_model.summary()

# ================================================================
# Step 7: Train MobileNetV2 Model
# ================================================================

print("\n" + "=" * 55)
print("TRAINING MOBILENETV2 TRANSFER LEARNING MODEL...")
print("=" * 55)

history_mobilenet = mobilenet_model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop]
)

# ================================================================
# Step 8: Evaluate Both Models on Test Set
# ================================================================

def evaluate_model(model, test_generator, name):
    """
    Predicts on test data and prints key metrics.
    Returns a dictionary of results.
    """
    test_generator.reset()

    # Get predicted probabilities
    preds = model.predict(test_generator, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()  # Convert to 0 or 1
    y_true = test_generator.classes                # True labels

    # Calculate metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"RESULTS: {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=['NORMAL', 'PNEUMONIA']))
    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'y_true': y_true, 'y_pred': y_pred
    }

# Run evaluation
res_cnn  = evaluate_model(custom_model,    test_data, "Custom CNN")
res_mnet = evaluate_model(mobilenet_model, test_data, "MobileNetV2 Transfer Learning")

# ================================================================
# Step 9: Plot Training Curves and Confusion Matrices
# ================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Pneumonia Detection - Model Evaluation", fontsize=14, fontweight='bold')

# --- Custom CNN Plots ---
axes[0,0].plot(history_cnn.history['accuracy'],     label='Train')
axes[0,0].plot(history_cnn.history['val_accuracy'], label='Validation')
axes[0,0].set_title('Custom CNN - Accuracy')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(history_cnn.history['loss'],     label='Train',      color='blue')
axes[0,1].plot(history_cnn.history['val_loss'], label='Validation', color='orange')
axes[0,1].set_title('Custom CNN - Loss')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Loss')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

cm1 = confusion_matrix(res_cnn['y_true'], res_cnn['y_pred'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0,2],
            xticklabels=['NORMAL','PNEUMONIA'],
            yticklabels=['NORMAL','PNEUMONIA'])
axes[0,2].set_title('Custom CNN - Confusion Matrix')
axes[0,2].set_xlabel('Predicted'); axes[0,2].set_ylabel('Actual')

# --- MobileNetV2 Plots ---
axes[1,0].plot(history_mobilenet.history['accuracy'],     label='Train',      color='green')
axes[1,0].plot(history_mobilenet.history['val_accuracy'], label='Validation', color='red')
axes[1,0].set_title('MobileNetV2 - Accuracy')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Accuracy')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(history_mobilenet.history['loss'],     label='Train',      color='green')
axes[1,1].plot(history_mobilenet.history['val_loss'], label='Validation', color='red')
axes[1,1].set_title('MobileNetV2 - Loss')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Loss')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

cm2 = confusion_matrix(res_mnet['y_true'], res_mnet['y_pred'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1,2],
            xticklabels=['NORMAL','PNEUMONIA'],
            yticklabels=['NORMAL','PNEUMONIA'])
axes[1,2].set_title('MobileNetV2 - Confusion Matrix')
axes[1,2].set_xlabel('Predicted'); axes[1,2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[INFO] Plot saved as results.png")

# ================================================================
# Step 10: Final Comparison Table
# ================================================================

print("\n" + "=" * 55)
print("FINAL COMPARISON: Custom CNN vs MobileNetV2")
print("=" * 55)
print(f"{'Metric':<15} {'Custom CNN':>15} {'MobileNetV2':>15}")
print("-" * 47)
metrics = ['acc', 'prec', 'rec', 'f1']
labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for m, l in zip(metrics, labels):
    print(f"{l:<15} {res_cnn[m]*100:>14.2f}% {res_mnet[m]*100:>14.2f}%")
print("=" * 55)
print("\n[INFO] Done!")