import librosa
import numpy as np
import warnings

def load_audio(file_path: str, target_sr: int = 22050, duration: float = None, offset: float = 0.0) -> tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, resamples it to target_sr, 
    and normalizes amplitude to [-1.0, 1.0].
    
    Args:
        file_path: Path to the audio file.
        target_sr: Desired sampling rate (default 22050 restricts max freq to ~11kHz).
        duration: Only load up to this many seconds.
        offset: Start loading from this time offset (seconds).
        
    Returns:
        audio: 1D NumPy array of the processed audio signal.
        sr: The sample rate.
    """
    try:
        # librosa handles MP3/WAV, auto-converts to mono (mono=True by default) and resamples.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Using librosa.load gives us float32 [-1.0, 1.0] samples natively
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True, offset=offset, duration=duration)
            
        # Ensure normalization in case the audio max amplitude exceeds bounds
        if audio.size > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {file_path}: {str(e)}")
