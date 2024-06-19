# 02_04_challenge.py

# Challenge: Enhance the CNN model

# Instructions:
# In this exercise, you will enhance the CNN model by adding more convolutional and pooling layers
# and experimenting with different dropout rates. Your task is to modify the existing model structure
# and observe the changes in the model's performance.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure TensorFlow uses CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
tf.config.set_visible_devices([], 'GPU')

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data by scaling pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the labels of the dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Print the shapes of the datasets to verify transformations
print(f"X_train shape: {X_train.shape}")  # Should still be (50000, 32, 32, 3)
print(f"y_train shape after one-hot encoding: {y_train.shape}")  # Should be (50000, 10)
print(f"X_test shape: {X_test.shape}")  # Should still be (10000, 32, 32, 3)
print(f"y_test shape after one-hot encoding: {y_test.shape}")  # Should be (10000, 10)

# Display a sample of training images with their labels again
def display_images(images, labels, y_labels, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()
    for i in np.arange(0, rows * cols):
        index = np.random.randint(0, len(images))  # Randomly select an image index
        axes[i].imshow(images[index])
        axes[i].set_title(labels[np.argmax(y_labels[index])])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# Display images again to see the normalized versions
display_images(X_train, labels, y_train)

# TASK: Enhance the CNN model by adding more convolutional and pooling layers,
# and experimenting with different dropout rates.

# Function to create an enhanced CNN model
def create_enhanced_cnn_model():
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),  # Pooling layer to reduce dimensionality
        Dropout(0.2),  # Dropout layer to prevent overfitting

        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),  # Pooling layer
        Dropout(0.3),  # Dropout layer

        # TASK: Add another convolutional layer here
        # Conv2D(128, (3, 3), activation='relu', padding='same'),
        # MaxPooling2D((2, 2)),  # Pooling layer
        # Dropout(0.4),  # Dropout layer

        # Fully connected layer
        Flatten(),  # Flatten the 2D arrays into a 1D vector
        Dense(512, activation='relu'),  # Dense layer with ReLU activation
        Dropout(0.5),  # Dropout layer
        Dense(10, activation='softmax')  # Output layer with softmax activation for classification
    ])
    return model

# Create the enhanced CNN model
model = create_enhanced_cnn_model()

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to understand its architecture
model.summary()

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test data to get the loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Save the trained model to the output directory
model.save('../output/cifar10_enhanced_model.h5')

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
