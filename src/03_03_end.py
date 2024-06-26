# 03_03_end.py

# Continue from the previous code in 03_03_begin.py

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

# Define a simple CNN model with detailed explanations
def create_cnn_model():
    model = Sequential([
        # First convolutional layer
        # Applies 32 filters of size 3x3 to the input image.
        # Detects basic features such as edges and textures.
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        
        # Max-pooling layer
        # Reduces dimensionality, retains important features.
        MaxPooling2D((2, 2)),
        
        # Dropout layer
        # Prevents overfitting by not allowing the network to become too reliant on any single node.
        Dropout(0.2),

        # Second convolutional layer
        # Applies 64 filters of size 3x3 to the feature maps from the previous layer.
        # Detects more complex features by combining simpler ones detected in the previous layer.
        Conv2D(64, (3, 3), activation='relu'),
        
        # Max-pooling layer
        # Further reduces dimensionality.
        MaxPooling2D((2, 2)),
        
        # Dropout layer
        # Further prevents overfitting.
        Dropout(0.3),

        # Fully connected (dense) layer
        # Flattens the 2D feature maps into a 1D vector and then applies a dense layer with 128 neurons.
        # Combines all the detected features to make the final classification decision.
        Flatten(),
        Dense(128, activation='relu'),
        
        # Dropout layer
        Dropout(0.4),
        
        # Output layer
        # Has 10 neurons (one for each class) and uses the softmax activation function to output probabilities for each class.
        Dense(10, activation='softmax')
    ])
    return model

# Ensure the output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the plot directory within the output directory
plot_path = os.path.join(output_dir, 'plots')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Define the model path
model_path = os.path.join(output_dir, 'cifar10_system_model.h5')

# Check if the model already exists
if os.path.isfile(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded existing model from {model_path}")
else:
    # Create the CNN model
    model = create_cnn_model()

    # Compile the model
    # The optimizer is Adam, which is an efficient version of gradient descent.
    # The loss function is categorical crossentropy, which is appropriate for multi-class classification problems.
    # We also track accuracy as a metric.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    # This gives an overview of the model architecture, including the number of parameters in each layer.
    model.summary()

    # Train the model
    # We train the model using the training data (X_train and y_train) for 20 epochs.
    # An epoch is one complete pass through the training data.
    # We also validate the model using the test data (X_test and y_test) to see how well it generalizes to unseen data.
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    # Save the trained model to the output directory
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Plot the training and validation accuracy over epochs
    # This helps us visualize how the model's performance improves (or doesn't) over time.
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Save the plot to a file
    plot_file = os.path.join(plot_path, '03_03_end_system_model.png')
    plt.savefig(plot_file)
    print(f'Plot saved to {plot_file}')
    
    plt.show()  # Show the plot
    plt.close()  # Close the figure after showing it