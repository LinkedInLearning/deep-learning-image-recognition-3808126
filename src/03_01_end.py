# 03_01_end.py

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure TensorFlow uses CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
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
print(f"X_train shape: {X_train.shape}")  # Should be (50000, 32, 32, 3)
print(f"y_train shape after one-hot encoding: {y_train.shape}")  # Should be (50000, 10)
print(f"X_test shape: {X_test.shape}")  # Should be (10000, 32, 32, 3)
print(f"y_test shape after one-hot encoding: {y_test.shape}")  # Should be (10000, 10)

# Function to display a sample of images from the dataset
def display_images(images, labels, y_data, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()
    for i in np.arange(0, rows * cols):
        index = np.random.randint(0, len(images))
        axes[i].imshow(images[index])
        label_index = np.argmax(y_data[index])  # Get the index of the label
        axes[i].set_title(labels[label_index])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# Display a sample of training images with their labels
display_images(X_train, labels, y_train)

# Define a function to illustrate a simple CNN architecture
def illustrate_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.summary()

# Illustrate a simple CNN architecture
illustrate_simple_cnn()

# Define a function to explain convolution operation
def explain_convolution():
    # Create a simple 3x3 image with a single channel
    image = np.array([[0, 1, 2],
                      [2, 2, 0],
                      [1, 0, 1]], dtype=float).reshape(1, 3, 3, 1)

    # Define a simple 2x2 filter
    filter = np.array([[1, 0],
                       [0, -1]], dtype=float).reshape(2, 2, 1, 1)

    # Perform convolution operation using tf.nn.conv2d
    conv_output = tf.nn.conv2d(image, filter, strides=[1, 1, 1, 1], padding='VALID')

    print("Input Image:\n", image.reshape(3, 3))
    print("Filter:\n", filter.reshape(2, 2))
    print("Convolution Output:\n", conv_output.numpy().reshape(2, 2))

# Explain convolution operation
explain_convolution()

# Define a function to explain pooling operation
def explain_pooling():
    # Create a simple 4x4 image with a single channel
    image = np.array([[1, 2, 1, 0],
                      [0, 1, 3, 1],
                      [2, 2, 0, 0],
                      [1, 0, 1, 3]], dtype=float).reshape(1, 4, 4, 1)

    # Apply max pooling operation
    pool_layer = MaxPooling2D((2, 2))
    pool_output = pool_layer(image)

    print("Input Image:\n", image.reshape(4, 4))
    print("Max Pooling Output:\n", pool_output.numpy().reshape(2, 2))

# Explain pooling operation
explain_pooling()