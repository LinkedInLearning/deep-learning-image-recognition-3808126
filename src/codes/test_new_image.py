import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os

# Define the relative path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'emotion_model.h5')

# Load the trained model from the file
model = tf.keras.models.load_model(model_path)

# List of emotion labels (assuming the order in your dataset is consistent)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(48, 48), color_mode='grayscale')  # Load and resize image
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Path to the test image
test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'input', 'images', 'test', 'happy', '80.jpg')  # Replace with your actual test image path

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Make a prediction
predictions = model.predict(test_image)
predicted_emotion = emotion_labels[np.argmax(predictions)]

# Print the predicted emotion
print(f'Predicted emotion: {predicted_emotion}')