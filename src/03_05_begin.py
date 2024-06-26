# 03_05_begin.py

# Introduction to challenges in image recognition

# In this section, we'll discuss various challenges faced in the field of image recognition.
# These challenges include dealing with different lighting conditions, handling occlusions, 
# managing scale variations, addressing class imbalance, and dealing with inter-class similarity.

print("Challenges in Image Recognition:")
print("1. Dealing with Different Lighting Conditions")
print("2. Handling Occlusions")
print("3. Scale Variations")
print("4. Dealing with Class Imbalance")
print("5. Inter-class Similarity")

# Let's illustrate these points with some basic examples and comments.

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

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

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the output directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the plot directory within the output directory
plot_path = os.path.join(output_dir, 'plots')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Define a CNN model with detailed explanations for challenges
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    return model

# Create the CNN model
model = create_cnn_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with a smaller number of epochs for demonstration purposes
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Generate classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(plot_path, '03_05_confusion_matrix.png')
plt.savefig(conf_matrix_path)
print(f'Confusion matrix plot saved to {conf_matrix_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it

# Classification Report
class_report = classification_report(y_true, y_pred_classes, target_names=labels)
print("Classification Report:\n", class_report)

# Save the model if needed
model_path = os.path.join(output_dir, 'cifar10_challenge_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")

# Challenges Discussion