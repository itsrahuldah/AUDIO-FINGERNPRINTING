"""
demo.py - Audio Fingerprinting Demonstration & Evaluation Script
================================================================
Runs the full Shazam-style pipeline on synthetic and real audio,
produces publication-quality figures, and evaluates noise robustness.

Usage:
    python demo.py                    # Synthetic-only demo
    python demo.py --use-real-songs   # Also test against real DB songs

All figures saved to results/ directory.
"""

import os
import sys
import glob
import time
import random
import warnings
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for figure saving
import matplotlib.pyplot as plt

from src.audio_processing import load_audio
from src.spectrogram import compute_stft, get_spectrogram
from src.fingerprint import find_peaks, generate_hashes
from src.database import SongDatabase
from src.matcher import match_hashes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SR = 22050          # Sample rate (Hz)
N_FFT = 2048        # FFT window size
HOP = 1024          # Hop length (samples)
DURATION = 10.0     # Synthetic signal duration (seconds)
QUERY_DUR = 5.0     # Query clip duration (seconds)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 150,
})


# ===========================================================================
# Helper functions
# ===========================================================================

def make_synthetic_signal(duration=DURATION, sr=SR):
    """Create a multi-tone synthetic signal (A4 + A5 + E5 with vibrato)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz (A4) + 880 Hz (A5) + 659.25 Hz (E5) with slight vibrato
    vibrato = 5.0 * np.sin(2 * np.pi * 6.0 * t)
    signal = (0.6 * np.sin(2 * np.pi * (440 + vibrato) * t) +
              0.3 * np.sin(2 * np.pi * 880 * t) +
              0.2 * np.sin(2 * np.pi * 659.25 * t))
    # Normalize
    signal = signal / np.max(np.abs(signal))
    return signal, t


def add_noise(signal, snr_db):
    """Add white Gaussian noise at given SNR (dB)."""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return np.clip(signal + noise, -1.0, 1.0)


def fingerprint_audio(audio):
    """Full pipeline: audio -> (peaks, hashes)."""
    stft = compute_stft(audio, n_fft=N_FFT, hop_length=HOP)
    spec = get_spectrogram(stft)
    peaks = find_peaks(spec)
    hashes = generate_hashes(peaks)
    return stft, spec, peaks, hashes


# ===========================================================================
# Visualization functions
# ===========================================================================

def plot_waveform(signal, t, save_path):
    """Plot the time-domain waveform."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t[:2000], signal[:2000], color='#58a6ff', linewidth=0.6, alpha=0.9)
    ax.fill_between(t[:2000], signal[:2000], alpha=0.15, color='#58a6ff')
    ax.set_title("Synthetic Audio Waveform (first 2000 samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_spectrogram_with_peaks(spec, peaks, save_path):
    """Plot spectrogram with constellation map overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Raw spectrogram
    im = axes[0].imshow(spec, aspect='auto', origin='lower', cmap='magma',
                        vmin=-60, vmax=np.max(spec))
    axes[0].set_title("Log-Magnitude Spectrogram")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Frequency Bin")
    axes[0].set_ylim(0, min(300, spec.shape[0]))
    plt.colorbar(im, ax=axes[0], format='%+2.0f dB', shrink=0.8)

    # Right: Spectrogram + constellation
    axes[1].imshow(spec, aspect='auto', origin='lower', cmap='magma',
                   vmin=-60, vmax=np.max(spec), alpha=0.6)
    if peaks:
        freqs, times = zip(*peaks)
        axes[1].scatter(times, freqs, c='#39d353', s=4, marker='o', alpha=0.8,
                        edgecolors='none')
    axes[1].set_title(f"Constellation Map ({len(peaks)} peaks)")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Frequency Bin")
    axes[1].set_ylim(0, min(300, spec.shape[0]))

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_hash_pairs(peaks, save_path, max_anchors=30):
    """Visualize anchor -> target zone connections."""
    fig, ax = plt.subplots(figsize=(12, 5))

    if peaks:
        freqs, times = zip(*peaks)
        ax.scatter(times, freqs, c='#8b949e', s=6, alpha=0.4, edgecolors='none',
                   label='Constellation Points')

    # Draw connections for a subset of anchors
    colors = ['#f78166', '#d2a8ff', '#58a6ff', '#39d353', '#e3b341']
    drawn = 0
    for i in range(min(len(peaks), 200)):
        if drawn >= max_anchors:
            break
        freq1, time1 = peaks[i]
        target_drawn = 0
        for j in range(1, 11):
            if i + j < len(peaks):
                freq2, time2 = peaks[i + j]
                delta_t = time2 - time1
                if 1 <= delta_t <= 200 and abs(freq2 - freq1) <= 200:
                    color = colors[drawn % len(colors)]
                    ax.plot([time1, time2], [freq1, freq2], color=color,
                            linewidth=0.8, alpha=0.6)
                    target_drawn += 1
        if target_drawn > 0:
            ax.scatter(peaks[i][1], peaks[i][0], c='#f85149', s=30,
                       zorder=5, edgecolors='white', linewidth=0.5)
            drawn += 1

    ax.set_title("Combinatorial Hash Pairs (Anchor → Target Zone)")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")
    ax.set_ylim(0, min(300, max(f for f, _ in peaks) + 20) if peaks else 300)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_offset_histogram(query_hashes, db, matched_song_id, save_path):
    """Plot the scatterplot and offset histogram for the best match."""
    from collections import defaultdict

    song_pairs = defaultdict(list)  # song_id -> [(query_t, db_t)]
    for h_val, q_off in query_hashes:
        matches = db.query(h_val)
        for (sid, db_off) in matches:
            song_pairs[sid].append((q_off, db_off))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatterplot for matched song
    if matched_song_id and matched_song_id in song_pairs:
        pairs = song_pairs[matched_song_id]
        q_times = [p[0] for p in pairs]
        d_times = [p[1] for p in pairs]
        deltas = [d - q for q, d in pairs]

        axes[0].scatter(q_times, d_times, c='#58a6ff', s=8, alpha=0.7,
                        edgecolors='none')
        axes[0].set_title(f"Scatterplot: Query vs DB Time (Match)")
        axes[0].set_xlabel("Query Time (frames)")
        axes[0].set_ylabel("Database Time (frames)")
        axes[0].grid(True, alpha=0.2)

        # Histogram of deltas
        axes[1].hist(deltas, bins=max(20, len(set(deltas)) // 2),
                     color='#39d353', alpha=0.8, edgecolor='#0d1117')
        axes[1].set_title("Offset Histogram (δt = db_time − query_time)")
        axes[1].set_xlabel("Time Offset δt (frames)")
        axes[1].set_ylabel("Count (Votes)")
        axes[1].grid(True, alpha=0.2)
    else:
        axes[0].text(0.5, 0.5, "No match found", ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=14)
        axes[1].text(0.5, 0.5, "No match found", ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=14)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_snr_results(snr_levels, results, save_path):
    """Bar chart of recognition rate vs SNR."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = []
    for r in results:
        if r >= 80:
            colors.append('#39d353')
        elif r >= 50:
            colors.append('#e3b341')
        else:
            colors.append('#f85149')

    bars = ax.bar([str(s) for s in snr_levels], results, color=colors,
                  edgecolor='#30363d', linewidth=0.8, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold',
                fontsize=10, color='#c9d1d9')

    ax.set_title("Recognition Rate vs Signal-to-Noise Ratio")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Recognition Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ===========================================================================
# Main demo routines
# ===========================================================================

def run_synthetic_demo():
    """Full demo with synthetic audio."""
    print("\n" + "=" * 60)
    print("  SYNTHETIC AUDIO DEMO")
    print("=" * 60)

    # 1. Generate signal
    print("\n[1/6] Generating synthetic multi-tone signal...")
    signal, t = make_synthetic_signal()
    plot_waveform(signal, t, os.path.join(RESULTS_DIR, "01_waveform.png"))

    # 2. Fingerprint the full signal
    print("[2/6] Computing STFT, spectrogram, and constellation map...")
    stft, spec, peaks, hashes = fingerprint_audio(signal)
    print(f"       Spectrogram shape: {spec.shape}")
    print(f"       Peaks detected: {len(peaks)}")
    print(f"       Hashes generated: {len(hashes)}")

    plot_spectrogram_with_peaks(spec, peaks,
                                os.path.join(RESULTS_DIR, "02_spectrogram_constellation.png"))
    plot_hash_pairs(peaks, os.path.join(RESULTS_DIR, "03_hash_pairs.png"))

    # 3. Build a mini database
    print("[3/6] Building in-memory database...")
    db = SongDatabase()
    db.add_song("Synthetic_MultiTone", hashes)

    # Also add a second distinct tone for discrimination testing
    t2 = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    signal2 = np.sin(2 * np.pi * 261.63 * t2)  # Middle C
    signal2 = signal2 / np.max(np.abs(signal2))
    _, _, _, hashes2 = fingerprint_audio(signal2)
    db.add_song("Synthetic_MiddleC", hashes2)
    print(f"       Database contains {db.next_song_id - 1} songs")

    # 4. Query with a noisy clip
    print("[4/6] Testing query with noisy 5-second clip...")
    start_sample = int(3.0 * SR)
    end_sample = int(3.0 * SR + QUERY_DUR * SR)
    query_clean = signal[start_sample:end_sample]
    query_noisy = add_noise(query_clean, snr_db=6)

    _, _, _, q_hashes = fingerprint_audio(query_noisy)
    best_id, best_offset, confidence = match_hashes(q_hashes, db)

    song_name = db.songs.get(best_id, "Unknown") if best_id else "No match"
    print(f"       Query hashes: {len(q_hashes)}")
    print(f"       MATCH: '{song_name}' (ID={best_id})")
    print(f"       Confidence (votes): {confidence}")
    if best_offset is not None:
        print(f"       Time offset: {best_offset} frames ~ {best_offset * HOP / SR:.2f}s")
    else:
        print(f"       Time offset: N/A")

    plot_offset_histogram(q_hashes, db, best_id,
                          os.path.join(RESULTS_DIR, "04_offset_histogram.png"))

    # 5. SNR robustness evaluation
    print("[5/6] Evaluating noise robustness across SNR levels...")
    snr_levels = [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
    num_trials = 20
    recognition_rates = []

    for snr in snr_levels:
        correct = 0
        for trial in range(num_trials):
            # Random crop position
            max_start = len(signal) - int(QUERY_DUR * SR)
            if max_start <= 0:
                start = 0
            else:
                start = random.randint(0, max_start)
            clip = signal[start:start + int(QUERY_DUR * SR)]
            noisy_clip = add_noise(clip, snr_db=snr)

            _, _, _, qh = fingerprint_audio(noisy_clip)
            sid, _, conf = match_hashes(qh, db)
            if sid == 1:  # Correct match to Synthetic_MultiTone
                correct += 1

        rate = 100.0 * correct / num_trials
        recognition_rates.append(rate)
        print(f"       SNR = {snr:+3d} dB  ->  {rate:5.1f}% ({correct}/{num_trials})")

    plot_snr_results(snr_levels, recognition_rates,
                     os.path.join(RESULTS_DIR, "05_snr_robustness.png"))

    # 6. Timing analysis
    print("[6/6] Timing analysis...")
    times_fingerprint = []
    times_match = []

    for _ in range(10):
        clip = signal[:int(QUERY_DUR * SR)]
        noisy = add_noise(clip, snr_db=6)

        t0 = time.perf_counter()
        _, _, _, qh = fingerprint_audio(noisy)
        t1 = time.perf_counter()
        times_fingerprint.append(t1 - t0)

        t0 = time.perf_counter()
        match_hashes(qh, db)
        t1 = time.perf_counter()
        times_match.append(t1 - t0)

    print(f"       Avg fingerprint time: {1000*np.mean(times_fingerprint):.1f} ms")
    print(f"       Avg match time:       {1000*np.mean(times_match):.3f} ms")
    print(f"       Total per query:      {1000*(np.mean(times_fingerprint)+np.mean(times_match)):.1f} ms")

    return snr_levels, recognition_rates


def run_real_song_demo():
    """Test against real songs in the database."""
    print("\n" + "=" * 60)
    print("  REAL SONG DATABASE DEMO")
    print("=" * 60)

    db_path = 'data/database/shazam_db.pkl'
    if not os.path.exists(db_path):
        print("  [SKIP] No database found. Run setup_dataset.py first.")
        return

    db = SongDatabase()
    db.load(db_path)
    print(f"\n  Loaded database with {db.next_song_id - 1} tracks")

    # Find query files
    query_files = glob.glob('data/queries/query_*.wav')
    if not query_files:
        # Generate queries from processed songs
        song_files = glob.glob('data/processed_songs/*.wav')
        if not song_files:
            print("  [SKIP] No processed songs found.")
            return
        print(f"  No pre-built queries; testing with {min(5, len(song_files))} random song clips...")
        test_files = random.sample(song_files, min(5, len(song_files)))
    else:
        test_files = query_files[:10]
        print(f"  Found {len(query_files)} query files, testing {len(test_files)}...")

    correct = 0
    total = 0

    for fpath in test_files:
        fname = os.path.basename(fpath)
        try:
            audio, sr = load_audio(fpath, target_sr=SR, duration=8.0)
            if 'query_' in fname:
                # Query files already have noise
                pass
            else:
                # Take a random 5s clip and add noise
                max_start = len(audio) - int(5.0 * SR)
                if max_start > 0:
                    start = random.randint(0, max_start)
                    audio = audio[start:start + int(5.0 * SR)]
                audio = add_noise(audio, snr_db=6)

            _, _, _, q_hashes = fingerprint_audio(audio)
            sid, offset, conf = match_hashes(q_hashes, db)

            matched_name = db.songs.get(sid, "???") if sid else "NO MATCH"
            expected = fname.replace("query_", "").replace(".wav", "")

            is_correct = expected in matched_name or matched_name in expected
            if is_correct:
                correct += 1
            total += 1

            status = "OK" if is_correct else "XX"
            print(f"    {status} '{fname}' -> '{matched_name}' (conf={conf})")
        except Exception as e:
            print(f"    XX '{fname}' -> ERROR: {e}")
            total += 1

    if total > 0:
        print(f"\n  Overall: {correct}/{total} correct ({100*correct/total:.0f}%)")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shazam Audio Fingerprinting Demo")
    parser.add_argument('--use-real-songs', action='store_true',
                        help='Also test against the real song database')
    args = parser.parse_args()

    print("+----------------------------------------------------------+")
    print("|   Audio Fingerprinting System - Demo & Evaluation        |")
    print("|   Based on: Wang, 'An Industrial-Strength Audio          |")
    print("|   Search Algorithm' (Shazam, 2003)                       |")
    print("+----------------------------------------------------------+")

    snr_levels, recognition_rates = run_synthetic_demo()

    if args.use_real_songs:
        run_real_song_demo()

    print("\n" + "=" * 60)
    print("  ALL DONE - Figures saved to results/")
    print("=" * 60)
