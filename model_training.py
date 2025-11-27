"""
Face Emotion Detection - Model Training Script
Assignment 2: Bioinformatics Masters Program

This script trains a CNN model to recognize 7 emotions from facial images.
Emotions: angry, disgust, fear, happy, neutral, sad, surprise
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

print("="*60)
print("FACE EMOTION DETECTION - MODEL TRAINING")
print("="*60)

# ============================================
# STEP 1: CONFIGURATION
# ============================================

print("\n[1/6] Setting up configuration...")

# Image parameters
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
EPOCHS = 30

# Data directories
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# Emotion labels (7 emotions)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

print(f"✓ Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"✓ Batch size: {BATCH_SIZE}")
print(f"✓ Training epochs: {EPOCHS}")
print(f"✓ Number of emotions: {NUM_CLASSES}")
print(f"✓ Emotions: {', '.join(EMOTIONS)}")

# ============================================
# STEP 2: DATA PREPARATION
# ============================================

print("\n[2/6] Preparing data generators...")

# Data augmentation for training (helps model learn better)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=10,           # Randomly rotate images
    width_shift_range=0.1,       # Randomly shift images horizontally
    height_shift_range=0.1,      # Randomly shift images vertically
    horizontal_flip=True,        # Randomly flip images
    zoom_range=0.1,              # Randomly zoom images
    validation_split=0.2         # Use 20% of training data for validation
)

# Test data (no augmentation, just normalize)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load training data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load test data
print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"\n✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")

# ============================================
# STEP 3: BUILD THE MODEL (CNN Architecture)
# ============================================

print("\n[3/6] Building the CNN model...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output layer (7 emotions)
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model architecture created!")
print(f"✓ Total parameters: {model.count_params():,}")

# Display model summary
print("\nModel Summary:")
model.summary()

# ============================================
# STEP 4: CALLBACKS (Training Helpers)
# ============================================

print("\n[4/6] Setting up training callbacks...")

# Early stopping - stops training if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when learning plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stop, reduce_lr]

print("✓ Callbacks configured!")

# ============================================
# STEP 5: TRAIN THE MODEL
# ============================================

print("\n[5/6] Starting model training...")
print("This will take 10-20 minutes depending on your computer!")
print("You'll see progress for each epoch...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================
# STEP 6: EVALUATE AND SAVE MODEL
# ============================================

print("\n[6/6] Evaluating model on test data...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n{'='*60}")
print(f"FINAL RESULTS:")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"{'='*60}")

# Save the trained model
MODEL_PATH = 'face_emotionModel.h5'
print(f"\nSaving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print(f"✓ Model saved successfully!")

# ============================================
# STEP 7: PLOT TRAINING HISTORY (OPTIONAL)
# ============================================

print("\n[7/7] Generating training history plots...")

try:
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✓ Training plots saved as 'training_history.png'")
    
except Exception as e:
    print(f"Note: Could not generate plots - {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nFiles created:")
print(f"  ✓ face_emotionModel.h5 - Trained model ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)")
print(f"  ✓ training_history.png - Accuracy/Loss graphs")
print("\nNext steps:")
print("  1. Test the model with sample images")
print("  2. Build the Flask web application")
print("  3. Integrate model with the web app")
print("="*60)

print("\n🎉 Model training successful! Ready for deployment!")