import os
# Force TensorFlow to use CPU and disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Directories for training and test datasets
TRAIN_DIR = 'input/images/train'
TEST_DIR = 'input/images/test'

# Data augmentation and normalization for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20%
    height_shift_range=0.2,  # Randomly shift images vertically by 20%
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

# Data normalization for validation images (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Generator to load and preprocess training images from directory
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=32,  # Load 32 images at a time
    color_mode='grayscale',  # Convert images to grayscale
    class_mode='categorical'  # Use categorical labels (one-hot encoding)
)

# Generator to load and preprocess validation images from directory
val_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=32,  # Load 32 images at a time
    color_mode='grayscale',  # Convert images to grayscale
    class_mode='categorical'  # Use categorical labels (one-hot encoding)
)

# Building the Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # First convolutional layer with 32 filters and 3x3 kernel
    MaxPooling2D(2, 2),  # First max pooling layer with 2x2 pool size
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer with 64 filters and 3x3 kernel
    MaxPooling2D(2, 2),  # Second max pooling layer with 2x2 pool size
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer with 128 filters and 3x3 kernel
    MaxPooling2D(2, 2),  # Third max pooling layer with 2x2 pool size
    Flatten(),  # Flatten the 3D output to 1D tensor for fully connected layers
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dropout(0.5),  # Dropout layer for regularization to prevent overfitting
    Dense(7, activation='softmax')  # Output layer with 7 units for 7 classes, using softmax activation
])

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data and validate using validation data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of steps per epoch
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,  # Number of validation steps
    epochs=50  # Train for 50 epochs
)

# Save the trained model to a file
model.save('emotion_model.h5')

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
