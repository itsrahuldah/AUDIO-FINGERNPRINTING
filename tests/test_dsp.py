import numpy as np
import os
import unittest
from src.spectrogram import compute_stft, get_spectrogram
from src.fingerprint import find_peaks, generate_hashes
from src.database import SongDatabase
from src.matcher import match_hashes

class TestDSPValidation(unittest.TestCase):
    def test_stft_sine(self):
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz Sine wave
        f_target = 440.0
        audio = np.sin(2 * np.pi * f_target * t)
        
        # Add 10dB noise to check robustness
        # Signal power approx 0.5
        noise_power = 0.5 / (10 ** (10 / 10))
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(t))
        audio_noisy = audio + noise
        
        n_fft = 2048
        hop_length = 1024
        stft_matrix = compute_stft(audio_noisy, n_fft=n_fft, hop_length=hop_length)
        spec_db = get_spectrogram(stft_matrix)
        
        # Verify 440 Hz is the peak column-wise
        # Freq resolution = sr / n_fft = 22050 / 2048 = 10.76 Hz/bin
        expected_bin = int(round(f_target / (sr / n_fft)))
        
        # Average power across frames
        avg_power = np.mean(spec_db, axis=1)
        max_bin = np.argmax(avg_power)
        
        self.assertTrue(abs(max_bin - expected_bin) <= 1, 
                        f"Expected bin {expected_bin}, got {max_bin}")

        # Test peak detection
        peaks = find_peaks(spec_db, fp_dim=15, time_dim=15, threshold=0)
        # There should be lots of peaks, but 440Hz bin should be picked up heavily
        freq_bins = [p[0] for p in peaks]
        self.assertTrue(any(abs(f - expected_bin) <= 1 for f in freq_bins))

    def test_fingerprint_shift_invariance(self):
        # Create a tiny constellation map (fake peaks)
        peaks_orig = [(100, 5), (150, 7), (200, 10), (120, 15)]
        hashes_orig = generate_hashes(peaks_orig, target_zone_fw=20, target_zone_freq=100, fan_out=5, min_time_delta=1)
        
        # Shift time by 50 frames
        shift = 50
        peaks_shifted = [(f, t + shift) for f, t in peaks_orig]
        hashes_shifted = generate_hashes(peaks_shifted, target_zone_fw=20, target_zone_freq=100, fan_out=5, min_time_delta=1)
        
        # The hash values MUST be identical, but the absolute times are shifted by 50.
        self.assertGreater(len(hashes_orig), 0)
        for (h1, t1), (h2, t2) in zip(hashes_orig, hashes_shifted):
            self.assertEqual(h1, h2)
            self.assertEqual(t2 - t1, shift)

    def test_matcher_offset_alignment(self):
        db = SongDatabase()
        db.add_song("Song A", [(10101, 100), (20202, 110), (30303, 120)])
        db.add_song("Song B", [(10101, 500), (99999, 510)])
        
        # Query hashes represent a crop from Song A starting at frame 10
        # So query times: 90, 100, 110
        query_hashes = [(10101, 90), (20202, 100), (30303, 110)]
        
        best_song_id, best_offset, max_votes = match_hashes(query_hashes, db)
        
        # Song A id is 1. Expected offset = db_offset (100) - query_offset (90) = 10.
        self.assertEqual(best_song_id, 1)
        self.assertEqual(best_offset, 10)
        self.assertEqual(max_votes, 3)

if __name__ == '__main__':
    unittest.main()

