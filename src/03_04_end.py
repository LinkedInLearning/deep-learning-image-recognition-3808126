# 03_04_end.py

# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Evaluate the model on the test data to get the loss and accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")

    # Predict the classes for the test data
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Generate a classification report
    class_report = classification_report(y_true, y_pred, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    print("Classification Report:\n", class_report)

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], yticklabels=[
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the evaluated model to the output directory
    model.save(model_path)
    print(f"Saved model to {model_path}")
else:
    print(f"Model not found at {model_path}. Please ensure the model is trained and saved correctly.")