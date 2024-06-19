# 03_02_begin.py

# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.datasets import cifar10

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure TensorFlow uses CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Print the shapes of the datasets to verify loading
print(f"X_train shape: {X_train.shape}")  # Should be (50000, 32, 32, 3)
print(f"y_train shape: {y_train.shape}")  # Should be (50000, 1)
print(f"X_test shape: {X_test.shape}")  # Should be (10000, 32, 32, 3)
print(f"y_test shape: {y_test.shape}")  # Should be (10000, 1)