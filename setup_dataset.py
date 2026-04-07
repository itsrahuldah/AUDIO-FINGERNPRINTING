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

def generate_query_clip(audio: np.ndarray, sr: int, base_name: str, duration_sec: int = 8):
    """Crops a random 8-second segment, adds noise, and saves as a query clip."""
    total_samples = len(audio)
    clip_samples = duration_sec * sr
    
    if total_samples <= clip_samples:
        start_idx = 0
        clip_samples = total_samples
    else:
        start_idx = random.randint(0, total_samples - clip_samples)
        
    clip = audio[start_idx:start_idx + clip_samples]
    
    # Add Gaussian Noise (approx 10dB SNR)
    noise_power = 0.5 / (10 ** (10 / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(clip))
    noisy_clip = clip + noise
    
    # Strictly bound to [-1.0, 1.0] to prevent audio clipping during WAV write
    noisy_clip = np.clip(noisy_clip, -1.0, 1.0)
    
    dest_path = os.path.join('data/queries', f"query_{base_name}.wav")
    sf.write(dest_path, noisy_clip, sr)

def build_pipeline(source_dataset_dir: str):
    setup_directories()
    
    # ENGINEERING DETAIL: Random Sampling
    # Extracing fingerprints is O(N) but memory-heavy. Ingesting the complete 100k FMA 
    # dataset requires distributed cloud clusters and multi-GB DBs. A 200-song uniform 
    # subset rigorously validates DSP pipeline behavior without killing local RAM.
    songs_to_process = gather_random_songs(source_dataset_dir, num_songs=200)
    if not songs_to_process:
        logging.info("Exiting pipeline. Please provide a valid source folder containing MP3s.")
        return
        
    db = SongDatabase()
    db_path = 'data/database/shazam_db.pkl'
    if os.path.exists(db_path):
        try:
            db.load(db_path)
            logging.info(f"Loaded existing database containing {db.next_song_id - 1} tracks. Appending new songs.")
        except Exception as e:
            logging.warning(f"Could not load existing DB. Creating fresh one. Error: {e}")
    
    # Optional: copy original mp3s to data/songs/ 
    for file_path in songs_to_process:
        try:
            shutil.copy(file_path, 'data/songs/')
        except shutil.SameFileError:
            pass
            
    logging.info("Starting Batch Processing and DB Ingestion...")
    
    # We use tqdm for memory-friendly localized batching (processing 1 track safely before garbage collecting)
    for file_path in tqdm(songs_to_process, desc="Ingesting Tracks"):
        try:
            # 1. Pipeline Preprocessing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr, _, base_name = process_and_save(file_path)
            
            # 2. STFT Spectrogram
            stft_mat = compute_stft(audio)
            spec_db = get_spectrogram(stft_mat)
            
            # 3. Peak Constellation & Hashing
            peaks = find_peaks(spec_db)
            hashes = generate_hashes(peaks)
            
            # 4. Save into Database Index
            db.add_song(base_name, hashes)
            
            # Generate queries for ~15% of the database to ensure we have test clips
            if random.random() < 0.15:
                generate_query_clip(audio, sr, base_name)
                
        except Exception as e:
            logging.error(f"Failed processing '{file_path}': {e}")
            
    # Serialize the Inverted Index to disk
    db_path = 'data/database/shazam_db.pkl'
    db.save(db_path)
    logging.info(f"Database successfully serialized to '{db_path}'. Total tracks: {db.next_song_id - 1}.")
    logging.info("--- PIPELINE COMPLETE ---")
    logging.info("The system is now primed. Run matcher over files in data/queries/ to verify.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Shazam Dataset Ingestion Pipeline")
    parser.add_argument('--source', type=str, default='./fma_small', help="Path to raw Kaggle dataset root (e.g., ./fma_small)")
    args = parser.parse_args()
    
    build_pipeline(args.source)
