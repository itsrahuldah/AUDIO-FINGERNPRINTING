import os
import random
import glob
import logging
import numpy as np
import soundfile as sf
import warnings
import shutil
from tqdm import tqdm

from src.audio_processing import load_audio
from src.spectrogram import compute_stft, get_spectrogram
from src.fingerprint import find_peaks, generate_hashes
from src.database import SongDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """Defines the final robust project structure"""
    dirs = ['data/songs', 'data/processed_songs', 'data/queries', 'data/database']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logging.info("Project directories initialized.")

def gather_random_songs(source_dir: str, num_songs: int = 200) -> list[str]:
    """Finds all mp3 files in the source directory and selects a random subset."""
    search_pattern = os.path.join(source_dir, '**', '*.mp3')
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        logging.warning(f"No MP3 files found in '{source_dir}'. Please ensure the Kaggle dataset is unzipped here.")
        return []
    
    if len(all_files) > num_songs:
        selected = random.sample(all_files, num_songs)
    else:
        selected = all_files
        
    logging.info(f"Selected {len(selected)} songs from {len(all_files)} total files.")
    return selected

def process_and_save(file_path: str, target_sr: int = 22050):
    """
    Loads audio, converts to mono, resamples, normalizes, and saves as WAV.
    
    ENGINEERING DETAILS:
    - 22050 Hz Resampling: Captures up to ~11kHz (Nyquist theorem). Covers the 
      fundamental harmonic signatures of music while halving memory and processing 
      overhead compared to standard 44.1kHz.
    - Mono Conversion: Spatial stereo panning contains no structural musical identity.
      Mixing channels averages the total spectral energy for consistent fingerprinting.
    - Normalization: Bounding amplitude to [-1.0, 1.0] prevents arbitrary loud/quiet
      recordings from violating dynamic threshold checks in the Peak Detection phase.
    """
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    dest_path = os.path.join('data/processed_songs', f"{base_name}.wav")
    
    audio, sr = load_audio(file_path, target_sr=target_sr)
    
    # Save standardized processed wave
    sf.write(dest_path, audio, sr)
    return audio, sr, dest_path, base_name

