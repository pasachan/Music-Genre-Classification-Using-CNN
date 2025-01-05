import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
BASE_AUDIO_PATH = './audio/genres'  
BASE_OUTPUT_PATH = './audio/spectrograms'  
SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 256
MODEL_PATH = './music_genre_cnn_model.h5'  # Path to the trained model
TARGET_SIZE = (308, 775)  # Input size for the model

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess spectrograms
def preprocess_spectrogram(image_path, target_size=TARGET_SIZE):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Function to create spectrograms
def create_spectrogram(file_path, output_path):
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=10)
    X = librosa.stft(data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
    X_magnitude = np.abs(X)
    X_db = librosa.amplitude_to_db(X_magnitude)

    plt.figure(figsize=(10, 4))
    plt.axis('off')  # Hide axes for a clean look
    librosa.display.specshow(X_db, sr=SAMPLE_RATE, hop_length=HOP_SIZE, cmap='inferno')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


true_labels = []
predicted_labels = []

for genre_idx, genre in enumerate(GENRES):
    AUDIO_FOLDER_PATH = os.path.join(BASE_AUDIO_PATH, genre)
    OUTPUT_FOLDER_PATH = os.path.join(BASE_OUTPUT_PATH, genre)

    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    for filename in os.listdir(AUDIO_FOLDER_PATH):
        if filename.endswith(".wav"):
            try:
                file_path = os.path.join(AUDIO_FOLDER_PATH, filename)
                output_filename = f"{filename[:-4]}_spectrogram.png"
                output_path = os.path.join(OUTPUT_FOLDER_PATH, output_filename)

                create_spectrogram(file_path, output_path)

                spectrogram = preprocess_spectrogram(output_path)
                prediction = model.predict(spectrogram)
                predicted_genre = np.argmax(prediction)

                true_labels.append(genre_idx)
                predicted_labels.append(predicted_genre)

                print(f"Processed {filename}: True = {genre}, Predicted = {GENRES[predicted_genre]}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(GENRES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)

plt.figure(figsize=(12, 10))
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix for Music Genre Classification")
plt.show()
