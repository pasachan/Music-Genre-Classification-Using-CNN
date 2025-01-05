import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']  
BASE_AUDIO_PATH = 'audio/genres' 
BASE_OUTPUT_PATH = 'audio/spectrograms'  
SAMPLE_RATE = 22050
WINDOW_SIZE = 1024
HOP_SIZE = 256


for genre in GENRES:
    AUDIO_FOLDER_PATH = os.path.join(BASE_AUDIO_PATH, genre)
    OUTPUT_FOLDER_PATH = os.path.join(BASE_OUTPUT_PATH, genre)

    
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    
    for filename in os.listdir(AUDIO_FOLDER_PATH):
        if filename.endswith(".wav"):
            file_path = os.path.join(AUDIO_FOLDER_PATH, filename)
            try:
                
                data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=10)

                
                X = librosa.stft(data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
                X_magnitude = np.abs(X)
                X_db = librosa.amplitude_to_db(X_magnitude)

                
                plt.figure(figsize=(10, 4))
                plt.axis('off')  
                librosa.display.specshow(
                    X_db,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_SIZE,
                    cmap='inferno'
                )

                output_filename = f"{filename[:-4]}_spectrogram.png"
                output_path = os.path.join(OUTPUT_FOLDER_PATH, output_filename)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                print(f"Saved spectrogram for {filename} in {OUTPUT_FOLDER_PATH}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
