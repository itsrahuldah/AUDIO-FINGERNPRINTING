import numpy as np

def compute_stft(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 1024) -> np.ndarray:
    """
    Computes the Short-Time Fourier Transform (STFT) from scratch using NumPy.
    This implementation mimics librosa's STFT but makes the DSP math explicit.
    
    Args:
        audio: 1D NumPy array (mono audio).
        n_fft: FFT window size (determines frequency resolution).
        hop_length: Number of samples between successive frames.
        
    Returns:
        stft_matrix: 2D complex array of shape (1 + n_fft/2, num_frames).
    """
    # Hanning window prevents spectral leakage at frame boundaries
    window = np.hanning(n_fft)
    
    # Pad audio to center the frames correctly (like librosa.stft(center=True))
    pad_len = n_fft // 2
    padded_audio = np.pad(audio, pad_len, mode='constant')
    
    # Calculate number of frames
    num_frames = 1 + (len(padded_audio) - n_fft) // hop_length
    
    # We use rfft since the input signal is real. 
    # rfft returns 1 + n_fft//2 frequency bins.
    stft_matrix = np.zeros((1 + n_fft // 2, num_frames), dtype=np.complex64)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        frame = padded_audio[start:end] * window
        # Compute real FFT
        stft_matrix[:, i] = np.fft.rfft(frame, n=n_fft)
        
    return stft_matrix

def get_spectrogram(stft_matrix: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Converts a complex STFT matrix into a log-magnitude spectrogram (dB scale).
    
    Args:
        stft_matrix: 2D complex array from compute_stft.
        ref: Reference value for dB conversion.
        amin: Minimum amplitude to avoid log(0).
        
    Returns:
        spectrogram_db: 2D float array matching stft_matrix shape, in decibels.
    """
    # Magnitude
    magnitude = np.abs(stft_matrix)
    
    # Power and log amplitude (dB)
    # We add amin to prevent log10(0) issues
    spectrogram_db = 20 * np.log10(np.maximum(amin, magnitude) / ref)
    return spectrogram_db
