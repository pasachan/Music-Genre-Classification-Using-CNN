import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('./music_genre_cnn_model.h5')

# Define genre labels (ensure these match the labels used during training)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to preprocess a spectrogram image
def preprocess_spectrogram(image_path, target_size=(308, 775)):
    """
    Preprocess the spectrogram image to match the input shape of the model.
    """
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return image_array

# Function to predict genre
def predict_genre_from_spectrogram(image_path):
    """
    Predict the genre from a pre-generated spectrogram.
    """
    spectrogram = preprocess_spectrogram(image_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Calculate accuracy for each genre
accuracy_per_genre = []
total_images_per_genre = []

# Parent directory containing genre folders
SPECTROGRAMS_DIR = './audio/spectrograms/training'  # Replace with your spectrograms directory

for genre_index, genre in enumerate(GENRES):
    genre_folder = os.path.join(SPECTROGRAMS_DIR, genre)
    correct_predictions = 0
    total_images = 0

    for filename in os.listdir(genre_folder):
        if filename.endswith('.png'):  # Assuming spectrograms are PNGs
            file_path = os.path.join(genre_folder, filename)
            predicted_class, _ = predict_genre_from_spectrogram(file_path)
            if predicted_class == genre_index:  # Check if prediction matches the true label
                correct_predictions += 1
            total_images += 1

    # Calculate accuracy for this genre
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    accuracy_per_genre.append(accuracy)
    total_images_per_genre.append(total_images)

    print(f"Genre: {genre} - Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")

# Plot accuracy per genre
plt.figure(figsize=(10, 6))
plt.bar(GENRES, accuracy_per_genre, color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Per Genre')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
