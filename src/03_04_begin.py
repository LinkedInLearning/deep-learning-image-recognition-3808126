# 03_04_begin.py

# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

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

# Ensure the output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the model path
model_path = os.path.join(output_dir, 'cifar10_system_model.h5')

# Check if the model already exists
if os.path.isfile(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded existing model from {model_path}")
else:
    print(f"Model not found at {model_path}. Please ensure the model is trained and saved correctly.")