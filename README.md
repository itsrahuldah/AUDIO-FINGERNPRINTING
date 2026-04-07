# Audio Fingerprinting System

An implementation of the Shazam audio recognition algorithm based on the seminal paper:

> **"An Industrial-Strength Audio Search Algorithm"**  
> Avery Li-Chun Wang, Shazam Entertainment, Ltd.  
> *Presented at the International Society for Music Information Retrieval (ISMIR), 2003*

## Overview

This project implements a complete audio fingerprinting and recognition pipeline that can identify audio tracks from short, noisy samples. The system uses **spectrogram peak constellation maps** and **combinatorial hashing** to create robust, noise-resistant audio fingerprints.

## Algorithm Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Raw Audio   │────▶│  STFT        │────▶│  Peak          │────▶│ Combinatorial│
│  (WAV/MP3)   │     │  Spectrogram │     │  Detection     │     │ Hashing      │
└─────────────┘     └──────────────┘     │  (Constellation│     │ (Anchor +    │
                                          │   Map)         │     │  Target Zone)│
                                          └────────────────┘     └──────┬───────┘
                                                                        │
                        ┌──────────────┐     ┌──────────────┐          │
                        │  Result:     │◀────│  Histogram   │◀─────────┘
                        │  Song ID +   │     │  Voting &    │   Hash Lookup
                        │  Confidence  │     │  Scoring     │   in Inverted
                        └──────────────┘     └──────────────┘   Index DB
```

### Pipeline Stages

1. **Audio Preprocessing** (`src/audio_processing.py`)  
   Load audio files, convert to mono, resample to 22050 Hz, normalize amplitude to [-1.0, 1.0].

2. **Spectrogram Generation** (`src/spectrogram.py`)  
   Compute Short-Time Fourier Transform (STFT) from scratch using NumPy with Hanning windowing. Convert to log-magnitude (dB) scale.

3. **Peak Detection / Constellation Map** (`src/fingerprint.py`)  
   Find robust local maxima in the spectrogram using a 2D maximum filter. Peaks survive noise because they represent relative energy maxima, not absolute values.

4. **Combinatorial Hashing** (`src/fingerprint.py`)  
   Pair each anchor peak with nearby target peaks to form hash tokens: `(freq1, freq2, Δt)` packed into 32-bit integers. This provides ~30 bits of specificity per hash, yielding massive search acceleration.

5. **Database Storage** (`src/database.py`)  
   Store hashes in an inverted index: `hash_value → [(song_id, time_offset), ...]`. Supports serialization via pickle.

6. **Matching / Scoring** (`src/matcher.py`)  
   For each query hash, look up matching database entries. Build an offset histogram per candidate song: `δt = db_offset - query_offset`. A peak in this histogram indicates a match. The score is the number of temporally aligned hash matches.

## Project Structure

```
shazam/
├── src/
│   ├── __init__.py            # Package init
│   ├── audio_processing.py    # Audio I/O and preprocessing
│   ├── spectrogram.py         # Custom STFT implementation
│   ├── fingerprint.py         # Peak detection + hash generation
│   ├── database.py            # Inverted index database
│   └── matcher.py             # Offset histogram matching
├── tests/
│   └── test_dsp.py            # Unit tests (STFT, shift-invariance, matching)
├── results/                   # Generated figures from demo
├── demo.py                    # Demo & evaluation script
├── setup_dataset.py           # Batch ingestion pipeline
├── shazam_clone.ipynb         # Interactive Jupyter notebook walkthrough
├── REPORT.md                  # Academic report (5-7 pages)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-.git
cd AUDIO-FINGERNPRINTING-

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Demo (Synthetic Audio)
```bash
python demo.py
```
Generates publication-quality figures in `results/` and runs noise robustness evaluation.

### Full Demo (with Real Songs)
```bash
python demo.py --use-real-songs
```

### Building a Song Database
```bash
# Point to a directory containing MP3 files
python setup_dataset.py --source /path/to/mp3/collection
```

### Running Tests
```bash
python -m pytest tests/test_dsp.py -v
```

### Interactive Notebook
```bash
jupyter notebook shazam_clone.ipynb
```

## Results

The system achieves:
- **95-100% recognition rate** at SNR ≥ 0 dB with 5-second query clips
- **~50% recognition** at approximately -6 dB SNR (15-second clips)
- **Sub-second query time** including fingerprinting and matching
- **Robust to**: additive noise, GSM compression artifacts, EQ changes

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 22050 Hz sample rate | Captures up to ~11 kHz (Nyquist) — sufficient for music identification |
| 2048-point FFT | ~10 Hz/bin frequency resolution at 22050 Hz |
| 32-bit hash packing | `freq1(11b) \| freq2(11b) \| Δt(10b)` — fits in a single integer |
| Fan-out factor F=10 | Balances storage cost (10× more hashes) vs search speed (10000× faster) |
| Inverted index | O(1) hash lookup; enables millisecond-scale search times |

## Limitations

- **No pitch/tempo invariance**: The algorithm assumes the playback speed matches the database version exactly.
- **In-memory database**: Current implementation stores everything in RAM. Production systems would use distributed key-value stores.
- **No live recording support**: Designed for identifying pre-recorded tracks, not live performances.
- **Single-channel only**: Stereo spatial information is discarded during preprocessing.

## Citation

```bibtex
@inproceedings{wang2003industrial,
  title={An Industrial-Strength Audio Search Algorithm},
  author={Wang, Avery Li-Chun},
  booktitle={Proceedings of the 4th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2003},
  organization={Shazam Entertainment, Ltd.}
}
```

## References

1. Wang, A. L.-C. and Smith, J. O., III., WIPO publication WO 02/11123A2, 7 February 2002.
2. Haitsma, J. and Kalker, A., "A Highly Robust Audio Fingerprinting System", ISMIR 2002, pp. 107-115.
3. Yang, C., "MACS: Music Audio Characteristic Sequence Indexing For Similarity Retrieval", IEEE WASPAA, 2001.

## Video Explanation on YouTube
The video explanation of the Shazam algorithm can be accessed on YouTube here:<br>
https://youtu.be/BTwkqNeb3HA?si=WrsdnJ20B06Am7ef

## Author

**Rahul Prashanth**  
GitHub: [itsrahuldah](https://github.com/itsrahuldah)  

**Tasmay Kaushik Tokarkar** <br>
GitHub: [tasmay566](https://github.com/tasmay566)


## License

This project is for academic and educational purposes. The algorithm is based on publicly available research by Shazam Entertainment, Ltd.
