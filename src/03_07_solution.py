# 03_05_solution.py

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
# 1. Dealing with Different Lighting Conditions
# Simulate different lighting conditions by adjusting brightness and contrast using simple scaling.

def adjust_brightness_contrast(image, alpha=1.0, beta=0.0):
    return np.clip(alpha * image + beta, 0, 1)

bright_image = adjust_brightness_contrast(X_train[0], alpha=1.5)
dark_image = adjust_brightness_contrast(X_train[0], alpha=0.5)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(bright_image)
plt.title('Bright Image')
plt.subplot(1, 2, 2)
plt.imshow(dark_image)
plt.title('Dark Image')
lighting_conditions_path = os.path.join(plot_path, '03_05_lighting_conditions.png')
plt.savefig(lighting_conditions_path)
print(f'Lighting conditions plot saved to {lighting_conditions_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it

# 2. Handling Occlusions
# Simulate occlusions by adding a black rectangle to the image using NumPy.

def add_occlusion(image, x, y, width, height):
    occluded_image = image.copy()
    occluded_image[x:x+width, y:y+height, :] = 0
    return occluded_image

occluded_image = add_occlusion(X_train[0], 8, 8, 16, 16)

plt.imshow(occluded_image)
plt.title('Occluded Image')
occlusion_path = os.path.join(plot_path, '03_05_occluded_image.png')
plt.savefig(occlusion_path)
print(f'Occluded image plot saved to {occlusion_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it

# 3. Scale Variations
# Show examples of images at different scales using tf.image.

def rescale_image(image, scale):
    return tf.image.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))

scaled_image_1 = rescale_image(X_train[0], 0.5)
scaled_image_2 = rescale_image(X_train[0], 1.5)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(scaled_image_1.numpy())
plt.title('Scaled Down Image')
plt.subplot(1, 2, 2)
plt.imshow(scaled_image_2.numpy())
plt.title('Scaled Up Image')
scale_variations_path = os.path.join(plot_path, '03_05_scale_variations.png')
plt.savefig(scale_variations_path)
print(f'Scale variations plot saved to {scale_variations_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it

# 4. Dealing with Class Imbalance
# Show class distribution and techniques to address imbalance.

class_counts = np.sum(y_train, axis=0)
plt.bar(labels, class_counts)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
class_distribution_path = os.path.join(plot_path, '03_05_class_distribution.png')
plt.savefig(class_distribution_path)
print(f'Class distribution plot saved to {class_distribution_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it

# Techniques to handle class imbalance include oversampling, undersampling, and using class weights during training.

# 5. Inter-class Similarity
# Illustrate how similar classes can be confused by the model.

def display_similar_images(images, labels, class_name_1, class_name_2):
    class_indices_1 = [i for i, label in enumerate(labels) if label == class_name_1]
    class_indices_2 = [i for i, label in enumerate(labels) if label == class_name_2]
    
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[class_indices_1[i]])
        plt.title('Cat' if class_name_1 == 3 else 'Dog')
        plt.subplot(2, 5, i+6)
        plt.imshow(images[class_indices_2[i]])
        plt.title('Cat' if class_name_2 == 3 else 'Dog')
    plt.suptitle('Similarity between Cat and Dog Images')
    similar_images_path = os.path.join(plot_path, '03_05_similar_images.png')
    plt.savefig(similar_images_path)
    print(f'Similar images plot saved to {similar_images_path}')
    plt.show()  # Show the plot
    plt.close()  # Close the figure after showing it

# Example: Cat and Dog images
display_similar_images(X_train, np.argmax(y_train, axis=1), 3, 5)

# 6. Dealing with Noise in Images
# Simulate noisy environments by adding random noise to the images.

def add_noise(image, noise_factor=0.1):
    noisy_image = image + noise_factor * np.random.randn(*image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

noisy_image = add_noise(X_train[0])

plt.imshow(noisy_image)
plt.title('Noisy Image')
noisy_image_path = os.path.join(plot_path, '03_07_noisy_image.png')
plt.savefig(noisy_image_path)
print(f'Noisy image plot saved to {noisy_image_path}')
plt.show()  # Show the plot
plt.close()  # Close the figure after showing it
