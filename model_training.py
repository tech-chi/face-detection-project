import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# --- Configuration ---
CSV_FILE = 'fer2013.csv'
MODEL_FILE = 'face_emotionModel.h5'
IMG_WIDTH = 48
IMG_HEIGHT = 48
NUM_CLASSES = 7
NUM_EPOCHS = 40  # Increased epochs for better chance at >60%
BATCH_SIZE = 64

# --- 1. Load and Preprocess Data ---

def load_and_preprocess_data(csv_path):
    """
    Loads FER2013.csv and preprocesses it into train and validation sets.
    """
    print(f"Loading dataset from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_path}")
        print("Please make sure 'fer2013.csv' is in the FACE_DETECTION directory.")
        return None
    
    # Separate data into training and validation (PublicTest)
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    
    print("Preprocessing data...")
    
    # Process training data
    X_train = np.array(list(map(str.split, train_data['pixels'])), dtype='float32')
    X_train = X_train.reshape(X_train.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
    X_train = X_train / 255.0  # Normalize
    y_train = to_categorical(train_data['emotion'], NUM_CLASSES)
    
    # Process validation data
    X_val = np.array(list(map(str.split, val_data['pixels'])), dtype='float32')
    X_val = X_val.reshape(X_val.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
    X_val = X_val / 255.0  # Normalize
    y_val = to_categorical(val_data['emotion'], NUM_CLASSES)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return (X_train, y_train), (X_val, y_val)

# --- 2. Build the CNN Model ---

def build_model(input_shape, num_classes):
    """
    Builds a robust CNN model for FER.
    """
    print("Building CNN model...")
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flatten and Dense layers
        Flatten(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# --- 3. Main Execution ---

def main():
    # Load data
    data_load_result = load_and_preprocess_data(CSV_FILE)
    if data_load_result is None:
        return
        
    (X_train, y_train), (X_val, y_val) = data_load_result
    
    # Build model
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
    model = build_model(input_shape, NUM_CLASSES)
    
    # Callback to save the best model
    # This ensures we save the model with the highest validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
        verbose=1
    )
    
    print("\n--- Model Training Complete ---")
    
    # Find and print the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    print(f"Best validation accuracy achieved: {best_val_accuracy * 100:.2f}%")
    
    if best_val_accuracy >= 0.60:
        print(f"Target validation accuracy (>=60%) met. Model saved to {MODEL_FILE}")
    else:
        print(f"Target validation accuracy not met, but saving best model anyway to {MODEL_FILE}")

if __name__ == "__main__":
    main()
