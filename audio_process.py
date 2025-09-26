import numpy as np, pandas as pd
import soundfile as sf
import librosa
import librosa.display
from scipy.signal import butter, sosfilt
import os
import matplotlib.pyplot as plt
import gc
import pandas as pd
from collections import defaultdict

species_counters = defaultdict(int)

df = pd.read_csv("cleaned_data.csv")
ml_catalog = df['ML Catalog Number'].astype(str)
rename_to = dict(zip(ml_catalog, df['Scientific Name'].astype(str)))
years = dict(zip(ml_catalog, df['Year']))

# Load audio paths
audios = []
path = "dataset/"
output = "mel_spectrograms/"
for file in os.listdir(path):
    if file.endswith(".wav") and file.split('.wav')[0] in list(ml_catalog):
        audios.append(os.path.join(path, file))
        
        
n = len(audios)
print(n)
batch_size = 10000

def filter_non_frog(data, sr, l_limit=800, h_limit=2500):
    sos = butter(12, [l_limit, h_limit], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, data)

def generate_spectrogram(y_norm, sr, species_name, occ_count, chunk_idx, rec_year):
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    species_folder = os.path.join(output, safe_species)
    os.makedirs(species_folder, exist_ok=True)

    filename = f"{safe_species}_{occ_count}_{chunk_idx}_{rec_year}.png"
    png_path = os.path.join(species_folder, filename)
    if os.path.exists(png_path):
        print(f"Skipping {filename} (already exists)")
        return

    S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def save_audio_chunk(y_chunk, sr, species_name, occ_count, chunk_idx, rec_year):
    safe_species = species_name.replace(" ", "_").replace("/", "_")
    species_folder = os.path.join(output, safe_species)  
    os.makedirs(species_folder, exist_ok=True)

    filename = f"{safe_species}_{occ_count}_{chunk_idx}_{rec_year}.wav"
    chunk_path = os.path.join(species_folder, filename)
    if not os.path.exists(chunk_path):
        sf.write(chunk_path, y_chunk, sr)
    
    
chunk_duration = 10
sr=16000
chunk_size = chunk_duration * sr
count = 1
for audio in audios:
    base_name = os.path.splitext(os.path.basename(audio))[0]
    recording_year = years.get(base_name, "Unknown")

    species_name = rename_to.get(base_name, "Unknown")
    safe_species = species_name.replace(" ", "_").replace("/", "_")

    species_counters[safe_species] += 1
    occ_count = species_counters[safe_species]

    y, sr = librosa.load(audio)
    for idx, chunk in enumerate(range(0, len(y), chunk_size)):
        y_chunked = y[chunk : chunk + chunk_size]
        y_filtered = filter_non_frog(y_chunked, sr)
        y_trim, _ = librosa.effects.trim(y_filtered, top_db=30)
        if len(y_trim) == 0 or np.max(np.abs(y_trim)) == 0:
            continue
        y_norm = y_trim / np.max(np.abs(y_trim))

        generate_spectrogram(y_norm, sr, species_name, occ_count, idx +1, recording_year)
        save_audio_chunk(y_norm, sr, species_name, occ_count, idx +1, recording_year)
        del y_chunked, y_filtered, y_trim, y_norm
        print(f'Rec {count}, chunk {idx} converted')
        gc.collect()
    count += 1

        
    
# for i in range(0, n, batch_size):
    # batch = audios[i:min(n, i + batch_size)]
    # for file in batch:
    #     try:
    #         y, sr = librosa.load(file, sr=16000)
    #         y_filtered = filter_non_frog(y, sr)
    #         y_trim, _ = librosa.effects.trim(y_filtered, top_db=30)
    #         y_norm = y_trim / np.max(np.abs(y_trim))


    #         base = os.path.splitext(os.path.basename(file))[0]
    #         generate_spectrogram(y_norm, sr, base)

    #         print(f"Processed {base}")

    #     except Exception as e:
    #         print(f"Failed {file}: {e}")
        
    #     del y, y_filtered, y_trim, y_norm
    #     gc.collect()
