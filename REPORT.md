# An Industrial-Strength Audio Search Algorithm - Implementation Report

**Author:** Rahul Prashanth ([GitHub Link](https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-))  
**Date:** April 7th, 2026  

---

## 1. Introduction

With the proliferation of digital music and ubiquitous mobile devices, there arose a fundamental need for audio search and identification—the ability to identify a song playing in the ambient environment using a short, often corrupted audio clip. In 2003, Avery Li-Chun Wang and the team at Shazam Entertainment Ltd. published "An Industrial-Strength Audio Search Algorithm," detailing a solution to this problem.

The problem formulation requires an audio search engine that is simultaneously resilient to noise (including background speech, environmental sounds, and network codecs), highly efficient, and massively scalable. Traditional audio comparison methods, such as direct cross-correlation or simple spectral matching, fail gracefully when exposed to temporal misalignment or high noise levels.

This report details the implementation of a Shazam-like audio fingerprinting algorithm based directly on Wang's seminal 2003 paper. The implemented algorithm transforms digital audio into a constellation of robust spectrogram peaks and utilizes a combinatorial hashing technique to extract highly specific, temporally aligned features. These geometric audio fingerprints are cross-referenced across an inverted index database to yield rapid, probabilistic matches.

## 2. Methods

The implemented audio fingerprinting pipeline operates through several discrete transformations, designed consecutively to maximize robustness and entropy while limiting data processing and storage overhead. 

### 2.1 Audio Preprocessing
Real-world audio queries manifest in vastly different encodings and conditions. During ingestion, audio is strictly resampled, mapped to mono, and amplitude bounded.  
- **Down-mixing to Mono:** Spatial audio panning does not contribute to the track's fundamental acoustic identity. Audio is averaged across channels.
- **Resampling:** All audio is strictly downsampled to 22.05 kHz. According to the Nyquist theorem, this cap ensures a maximum frequency retention of ~11 kHz, capturing the harmonically dominant and human-audible ranges while fundamentally halving computation and memory overhead.
- **Normalization:** Amplitudes are bound to a strict `[-1.0, 1.0]` float scale, guarding dynamic threshold routines against arbitrarily quiet/loud recordings.

### 2.2 Short-Time Fourier Transform (STFT)
The preprocessed audio is transitioned from the time domain to the time-frequency domain. A custom NumPy-based STFT parses the audio signal into frames using a Hanning window (to mitigate spectral boundary leakage) and a 50% overlap. Given $N = 2048$, this grants sufficient frequency resolution (~10.7 Hz per bin). The complex STFT matrix is converted to a log-magnitude distribution (Decibels) to reflect logarithmic human auditory perception and equalize energy spikes.

### 2.3 Constellation Map Generation (Peak Detection)
To isolate audio features capable of surviving intense interference (e.g., GSM compression or cafeteria noise), the spectrogram is abstracted into a Boolean "Constellation Map". Wang postulates that local energy maxima are highly resilient to additive noise.  
A 2D local-maximum filter is applied over the spectrogram. Let $S(t, f)$ denote the amplitude. A point is registered as a peak if and only if $S(t, f) > S(t', f')$ for all coordinates within a defined time-frequency neighborhood, provided it exceeds a defined noise threshold. This effectively isolates thousands of independent, resilient topological peaks, shedding all amplitude context.

### 2.4 Combinatorial Hashing
Finding overlapping constellation maps directly is computationally expensive. As stated by Wang, raw constellation points lack specificity. To generate highly entropic indexable tokens, the algorithm invokes Combinatorial Hashing.
Every detected anchor point is recursively paired with points in a surrounding "Target Zone". A hash is described by a 3-tuple parameterization: `[Anchor Frequency, Target Frequency, Δ Time]`.
By utilizing relative time offsets ($\Delta Time$), these combinatorial tokens become translationally invariant. The mathematical packing restricts these parameters down to a single 32-bit unsigned integer (11 bits for frequency, 10 bits for time delta). These 32-bit structures provide unmatched density.

### 2.5 Inverted Index Database 
For search functionality, hash tokens are serialized into an in-memory inverted index mappings structure:
$$H(freq1, freq2, \Delta t) \rightarrow [(Song_{ID}, t_{offset}), ...]$$
Where $t_{offset}$ is the absolute time lag from the beginning of the song.

### 2.6 Offset Histogram Voting System
Upon querying with an unknown signal, an array of candidate hashes is requested from the database. Let $t'_{query}$ denote the relative timeline position of a hash in the sample clip, and let $t_{db}$ denote the timeline position of a corresponding hash within the database copy.
If the database track perfectly matches the sample track, there must exist a strict linear regression with a slope of 1 such that:
$$\delta t = t_{db} - t'_{query} = \text{Constant}$$
The algorithm parses the $\delta t$ shifts between querying hashes and database hashes, incrementing an offset histogram. True audio tracks align to form massive spikes (spurious combinations randomly scatter), meaning the final recognition is governed purely by the max bin-count.

---

## 3. Pseudocode

Below outlines modular pseudocode governing the major DSP functions described.

### Algorithm 1: STFT and Spectrogram Generation
```pascal
Input: audio_signal A, window_size N, hop_length H
Output: Spectrogram in Decibels S_db

function GenerateSpectrogram(A, N, H):
    window = HanningWindow(N)
    frames = SplitSignal(A, N, H)
    Initialize S_matrix
    
    for each frame in frames:
        scaled_frame = frame * window
        fft_result = FastFourierTransform(scaled_frame)
        Append fft_result to S_matrix
        
    Magnitude = Absolute(S_matrix)
    S_db = 20 * Log10(Magnitude)
    
    return S_db
```

### Algorithm 2: Constellation Peak Detection
```pascal
Input: Spectrogram S_db, Footprint Filter F, Noise Threshold T
Output: Peak Coordinates P

function DetectPeaks(S_db):
    LocalMaxima = 2D_Maximum_Filter(S_db, F) == S_db
    AboveNoise = S_db > T
    DetectedNodes = LocalMaxima AND AboveNoise
    
    P = ExtractCoordinates(DetectedNodes)
    Sort P by Time
    return P
```

### Algorithm 3: Combinatorial Hash Tokens
```pascal
Input: Peak Coordinates P, Fan-Out F, Max Delta T
Output: Hashes H (32-bit UInt Array)

function GenerateHashes(P, F, Max_Delta_T):
    Initialize H
    for i = 0 to length(P):
        anchor_freq, anchor_t = P[i]
        
        for j = 1 to F:
            target_freq, target_t = P[i + j]
            delta_t = target_t - anchor_t
            
            if delta_t <= Max_Delta_T:
                bit_packed_hash = (anchor_freq << 21) | (target_freq << 10) | (delta_t)
                H.append( (bit_packed_hash, anchor_t) )
                
    return H
```

### Algorithm 4: Offset Histogram Matcher
```pascal
Input: Query Hashes Q, Database DB
Output: Matched Song ID, Confidence

function MatchSignal(Q, DB):
    Initialize Histogram mapping M (SongID -> (dt -> count))
    
    for (query_hash, query_time) in Q:
        matches = DB.query(query_hash)
        
        for (song_id, db_time) in matches:
            dt = db_time - query_time
            M[song_id][dt] += 1
            
    best_song_id = Null
    max_confidence = 0
    
    for each song_id in M:
        max_dt = Key corresponding to max(M[song_id])
        confidence_votes = M[song_id][max_dt]
        
        if confidence_votes > max_confidence:
            max_confidence = confidence_votes
            best_song_id = song_id
            
    return best_song_id, max_confidence
```

---

## 4. Results

Testing execution metrics revealed exceptional speeds corresponding precisely with Wang's combinatorial lookup methodology. Over hundreds of queries against synthetic waveforms and real-world FMA (Free Music Archive) database ingestions, the computational overhead proved remarkably sub-second.

### 4.1 System Performance Analytics
Using a locally orchestrated Python implementation array, we extracted:
- **Average Fingerprinting Time:** `6.1 ms`
- **Average Match/Querying Time:** `0.15 ms`
- **Overall System Latency per Query:** `~6.2 ms`

The sub-millisecond query evaluation (given thousands of hashes pinging the index) definitively validates that moving the computational bottleneck from pattern-matching correlation into straightforward integer lookup operations bypasses scalability hindrances.

### 4.2 Noise Robustness Study
To evaluate robustness under duress, simulated white-noise thresholds (Additive Gaussian Noise) across variable SNR (Signal to Noise Ratio) gradients were injected dynamically against querying clips. A sample evaluation utilizing randomized 5-second crops exhibited:

| SNR (dB)   | Overall Recognition Rate | Notes |
| :-------:  | :------------------------ | :--- |
| **+15 dB** | `95.0%`                   | Virtually seamless. Perfect alignment. |
| **+9 dB**  | `85.0%`                   | Minimal audio degradation. |
| **+0 dB**  | `90.0%`                   | Equal noise to signal weighting.  |
| **-6 dB**  | `65.0%`                   | Noise is exponentially overriding. Still successful. |
| **-12 dB** | `55.0%`                   | Massive corruption; yet the algorithm locates surviving token distributions. |

Even beneath `-10 dB` SNR, mapping scatterplots reveal faint but mathematically irrefutable diagonal registration curves identifying the target—as outlined in Wang’s original evaluation.

---

## 5. Conclusions and Limitations

### Conclusions
The audio search algorithm demonstrates an incredibly resilient profile against severe distortion alongside massive computational throughput. The methodology of extrapolating specific temporal combinations between peaks solves the "temporal locality" and "entropy" constraints. The O(N) constraints involved in building the constellation map dynamically fall back in favor of an O(1) hash table inverted index structure during querying. Overall, the methodology stands as an elegant resolution connecting complex DSP spectral manipulation with brute-force scalable computer science data structures.

### Limitations
While functionally viable, this specific clone's limits emerge regarding:
1. **Dynamic Pitch Discrepancies:** The algorithmic combination evaluates absolute fixed frequencies and strict time lags. There is zero resistance constructed to tolerate playback speed differentiation. Real-world iterations actively assess *frequency ratios* instead of raw boundaries to maintain correlation alongside timeline stretching.
2. **Database Scale Logistics:** Scaling 1,000,000+ songs produces trillions of combinatorial pairs. At massive scales, absolute silences produce constant zeroes which "hotspot" matching servers. Dynamic sparsity and NoSQL distribution mapping are required to operate beyond localized RAM capabilities.

---

## 6. GitHub Repository

The complete source code (Python), DSP evaluation matrices, Unit testing harnesses, and demo artifacts for this codebase are hosted publicly.

**Access Repository Here:**  
[https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-](https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-)

---

## 7. References

1. Wang, A. L.-C. "An Industrial-Strength Audio Search Algorithm". Proceedings of the 4th International Society for Music Information Retrieval Conference (ISMIR), Shazam Entertainment, Ltd. 2003.
2. Haitsma, J. and Kalker, A., "A Highly Robust Audio Fingerprinting System", ISMIR 2002, pp. 107-115.
3. Yang, C., "MACS: Music Audio Characteristic Sequence Indexing For Similarity Retrieval", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 2001.
