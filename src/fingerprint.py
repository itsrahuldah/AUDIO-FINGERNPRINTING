import numpy as np
import scipy.ndimage as ndimage

def find_peaks(spectrogram_db: np.ndarray, 
               fp_dim: int = 15, 
               time_dim: int = 15, 
               threshold: float = 10.0) -> list[tuple[int, int]]:
    """
    Finds robust local maxima in the spectrogram (constellation map).
    """
    neighborhood = np.ones((fp_dim, time_dim), dtype=bool)
    
    # 2D max filter
    local_max = ndimage.maximum_filter(spectrogram_db, footprint=neighborhood) == spectrogram_db
    
    # Background threshold
    background = (spectrogram_db > threshold)
    
    # True peaks are local maxima above the noise floor
    detected_peaks = local_max & background
    
    # Get coordinates of peaks
    freq_idx, time_idx = np.where(detected_peaks)
    
    peaks = list(zip(freq_idx, time_idx))
    # Sort by time to facilitate forward-looking hash generation
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_hashes(peaks: list[tuple[int, int]], 
                    target_zone_fw: int = 200, 
                    target_zone_freq: int = 200,
                    fan_out: int = 10,
                    min_time_delta: int = 1) -> list[tuple[int, int]]:
    """
    Generates combinatoral hashes from the constellation map by pairing anchor peaks
    with points in a restricted target zone.
    """
    hashes = []
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        freq1, time1 = peaks[i]
        
        # Target Zone Fan-out limits combinatorial explosion
        for j in range(1, fan_out + 1):
            if i + j < num_peaks:
                freq2, time2 = peaks[i + j]
                delta_t = time2 - time1
                
                # Boundary constraints for target zone
                if min_time_delta <= delta_t <= target_zone_fw:
                    if abs(freq2 - freq1) <= target_zone_freq:
                        # Pack into a 32-bit int: freq1 (10 bits) | freq2 (10 bits) | delta_t (12 bits)
                        h = ((freq1 & 0x7FF) << 21) | ((freq2 & 0x7FF) << 10) | (delta_t & 0x3FF)
                        hashes.append((h, time1))
    return hashes
