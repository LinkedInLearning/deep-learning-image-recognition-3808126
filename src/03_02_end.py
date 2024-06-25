# 03_02_end.py

# Continue from the previous code in 03_02_begin.py

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Print the shapes of the datasets to verify transformations
print(f"X_train shape: {X_train.shape}")  # Should be (50000, 32, 32, 3)
print(f"y_train shape after one-hot encoding: {y_train.shape}")  # Should be (50000, 10)
print(f"X_test shape: {X_test.shape}")  # Should be (10000, 32, 32, 3)
print(f"y_test shape after one-hot encoding: {y_test.shape}")  # Should be (10000, 10)

# Define the output directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the plot directory within the output directory
plot_path = os.path.join(output_dir, 'plots')

# Create the directory if it doesn't exist
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Define a function to display a sample of images from the dataset
def display_images(images, y_data, rows=4, cols=4, title="Images", save_path=None):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(title)
    axes = axes.ravel()
    for i in np.arange(0, rows * cols):
        index = np.random.randint(0, len(images))
        axes[i].imshow(images[index])
        label_index = np.argmax(y_data[index])  # Get the index of the label
        axes[i].set_title(labels[label_index])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    plt.show()  # Show the plot
    plt.close(fig)  # Close the figure after showing it

# Define the file path to save the plot for original images
original_images_plot_file = os.path.join(plot_path, '03_02_original_images.png')

# Display a sample of training images with their labels and save the plot
display_images(X_train, y_train, title="Before Augmentation", save_path=original_images_plot_file)

# Image Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,      # Rotate images by up to 15 degrees
    width_shift_range=0.1,  # Shift images horizontally by up to 10%
    height_shift_range=0.1, # Shift images vertically by up to 10%
    horizontal_flip=True,   # Flip images horizontally
    zoom_range=0.2,         # Zoom into images by up to 20%
    channel_shift_range=0.1, # Randomly change the brightness
    shear_range=0.2         # Shear images by up to 20%
)

# Fit the data generator to the training data
datagen.fit(X_train)

# Define a function to visualize augmented images
def visualize_augmented_images(datagen, images, y_data, rows=4, cols=4, title="Augmented Images", save_path=None):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(title)
    axes = axes.ravel()
    for X_batch, y_batch in datagen.flow(images, y_data, batch_size=rows*cols):
        for i in np.arange(0, rows * cols):
            axes[i].imshow(X_batch[i])
            label_index = np.argmax(y_batch[i])  # Get the index of the label
            axes[i].set_title(labels[label_index])
            axes[i].axis('off')
        plt.subplots_adjust(hspace=0.5)
        if save_path:
            plt.savefig(save_path)
            print(f'Plot saved to {save_path}')
        plt.show()  # Show the plot
    plt.close(fig)  # Close the figure after showing it
        break  # We only want to visualize one batch

# Define the file path to save the plot for augmented images
augmented_images_plot_file = os.path.join(plot_path, '03_02_augmented_images.png')

# Visualize augmented images
visualize_augmented_images(datagen, X_train, y_train, title="After Augmentation", save_path=augmented_images_plot_file)