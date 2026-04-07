**Project Report**

**Software Implementation of Audio Fingerprinting (Shazam Algorithm)**

Tasmay Kaushik Tokarkar

GitHub: \[tasmay566\] (<https://github.com/tasmay566>)

Rahul Prashanth

GitHub: \[itsrahuldah\] (<https://github.com/itsrahuldah>)

**1. Introduction:**

Imagine you are in a noisy club, where many people are talking, dancing, singing and there is a song playing in the background. You really like the sound of it and you want to know which song it is. So in order to find out the song, you will most like have to rely on a software such as Shazam. Have you ever wondered how your phone can listen to a noisy room for just a few seconds and accurately name the song that is playing?

For this project, we decided to build a software implementation of the audio fingerprinting algorithm used by apps like Shazam using Python. At its core, sound is just a messy, one-dimensional wave of air pressure. The challenge of this project wasn\'t just matching audio files; it was taking that messy wave, extracting its unique mathematical DNA, which is called the fingerprint and building a search engine that could find a match in a database almost instantly, even if the recording was full of background noise. Here is how we broke the problem down and solved it.

**2. Methods:**

**2.1 Processing the Raw Audio**

Before we could identify a song, we needed to get the audio into a format Python could actually read. Whether we were analyzing a studio track for our database or a noisy 10-second recording from a phone microphone, we first converted the audio into a digital array of numbers. We mixed the stereo audio down to a single mono channel and sampled it at a standard rate so that every piece of audio we processed was speaking the exact same mathematical language.

**2.2 The Short-Time Fourier Transform (STFT)**

A raw audio wave just tells us how loud a sound is over time, which isn\'t very helpful for identifying a song. We needed to know what notes were playing. To do this, we passed our audio array through a Short-Time Fourier Transform (STFT). Think of STFT as a prism that splits white light into a rainbow. It chops our audio into tiny time slices and reveals exactly which frequencies (or pitches) are present in each slice. This transforms our 1D sound wave into a 2D \"Spectrogram\"---a visual representation where one axis is time, the other is frequency, and the brightness shows the intensity of the sound. It is essentially the digital sheet music of the song.

**2.3 Constellation Mapping (Finding the Peaks)**

A spectrogram contains millions of data points, which is way too much information to search through quickly. To simplify things, we applied a peak-finding algorithm. Instead of looking at the entire spectrogram, we only isolated the loudest, most dominant frequencies in a given neighborhood of time and pitch. We stripped away all the quiet background noise and kept only the strongest notes. When you plot these remaining points on a graph, it looks exactly like a map of the night sky, which is why it is called a \"Constellation Map.\"

**2.4 Combinatorial Hashing (Connecting the Dots)**

Here is where the real computer science comes in. A single dot on our constellation map (like a note playing at 500 Hz) isn\'t unique; millions of songs use that exact same note. To create a unique identifier, we looked at the relationship between the dots. We took a \"target\" dot and paired it with an \"anchor\" dot that happened a few moments later.

We took three specific pieces of information:

1.  The frequency of the first dot.

2.  The frequency of the second dot.

3.  The exact time difference (delta t) between them.

We packed these three numbers together into a single integer called a \"hash.\" Because the time difference between two notes never changes regardless of when you start recording, these hashes act as highly specific, unbreakable fingerprints for the song.

**2.5 Database Matching & Temporal Alignment**

Once we generated the hashes for our short recording, we threw them at our Python database (a dictionary mapping hashes to song IDs and timestamps) to see what matched. Because our hashes are relatively short, we got a lot of false positives---random background noise that accidentally matched random songs.

To find the true match, we performed a step called Temporal Alignment. For every matched hash, we subtracted the timestamp of where it occurred in our recording from the timestamp of where it occurred in the database track. This gave us a \"Time Offset.\"

For the false, random matches, these offset numbers were scattered all over the place. But for the correct song, the time offset remained perfectly constant, creating a massive spike in our data (or a perfect diagonal line if graphed). The algorithm simply looks for the song that produces this massive spike of identical offsets. The moment it finds that grouping, it confidently declares a match.

**3. Pseudo Codes:**

Below are the pseudocodes that govern the major DSP functions described.

**Algorithm 1: STFT and Spectrogram Generation**

![](media/image1.png){width="5.094460848643919in" height="4.021394356955381in"}

**Algorithm 2: Constellation Peak Detection**

![](media/image2.png){width="5.479931102362205in" height="2.7816382327209097in"}

**Algorithm 3: Combinatorial Hash Tokens**

![](media/image3.png){width="6.268055555555556in" height="3.2368055555555557in"}

**Algorithm 4: Offset Histogram Matcher**

![](media/image4.png){width="4.39915135608049in" height="4.769961723534558in"}

**4. Results:**

**4.1 Following are the screenshots of the various results obtained after each step in the audio processing:**

a\) Audio Waveform:

![](media/image5.png){width="6.268055555555556in" height="1.5125in"}

b\) Spectrogram and Constellation map:

![](media/image6.png){width="6.268055555555556in" height="2.223611111111111in"}

c\) Combinatorial Hash Pairs:

![](media/image7.png){width="6.268055555555556in" height="2.557638888888889in"}

d\) Offset Histogram:

![](media/image8.png){width="6.268055555555556in" height="2.189583333333333in"}

e\) Recognition Rate vs Signal-to-Noise Ratio

![](media/image9.png){width="6.268055555555556in" height="3.0930555555555554in"}

**4.2 System Performance Analytics**

After putting our Python code together, we had to see how well it actually worked. We ran hundreds of test queries using both computer-generated audio and real songs from the Free Music Archive. We saw that the algorithm is incredibly fast.

Here is exactly how long our system took to process a query:

-   **Creating the Fingerprint:** 6.1 milliseconds

-   **Searching the Database (The Lookup):** 0.15 milliseconds

-   **Total Time to Match:** \~6.2 milliseconds

Why is it fast? It all goes back to the hashing step we talked about earlier. Taking less than a millisecond to search the database proves that our strategy worked. Because we converted the audio into simple integer hashes, the computer isn\'t doing any heavy, slow pattern-matching. It is just doing a straightforward number lookup, bypassing all the usual bottlenecks.

**4.3 Testing with Background Noise**

In the real world, getting noiseless audio is very difficult. To see how much noise our code could handle, we took random 5-second audio clips and added artificial white noise.

We measured this using the Signal-to-Noise Ratio (SNR). Here is how our algorithm held up as things got louder:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Signal-to-Noise Ratio (SNR)**   **Recognition Rate**   **Notes**
  --------------------------------- ---------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **+15 dB**                        95.0%                  Virtually seamless- The background noise was barely there, and the algorithm aligned perfectly.

  **+9 dB**                         85.0%                  A little noisy. Just minor audio degradation, but still a very solid match rate.

  **+0 dB**                         90.0%                  Equal parts music and noise. The background noise was exactly as loud as the song, yet it still found the match almost every time.

  **-6 dB**                         65.0%                  The noise is taking over. Even with the noise being significantly louder than the music, we still successfully identified the song well over half the time.

  **-12 dB**                        55.0%                  Massive corruption. The audio sounded like total garbage. But incredibly, the algorithm still managed to find enough surviving \"dots\" to get a match 55% of the time.
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Even beneath -10 dB SNR, mapping scatterplots reveal faint but mathematically irrefutable diagonal registration curves identifying the target---as outlined in Wang's original evaluation.

**5. Conclusions:**

To summarize, this project proved that the audio fingerprinting algorithm is both incredibly tough against heavy distortion and it is lightning-fast.

By looking at the time combinations between peaks instead of just isolated dots, we solved the problem of trying to search for generic, repeating data. The real beauty of this algorithm is how it handles the workload. We took a heavy, time-consuming math problem (building the constellation map) and turned the final search into an instant O(1) hash table lookup. It is an elegant bridge between complex Digital Signal Processing (DSP) and raw Computer Science.

**Limitations of Our Python Clone and Future prospect:**

While our code works brilliantly for a localized database, it does have a few distinct limits compared to the real-world app:

1.  **Changes in Speed or Pitch:** Our algorithm looks for absolute, fixed frequencies and strict time gaps. This means if you play a song even slightly faster or slower (like a DJ mixing a track) our code will completely fail to find a match. The real-world Shazam algorithm fixes this by looking at frequency ratios instead of raw numbers, so the fingerprint survives even if the audio is stretched. We are planning to solve this exact problem in the future.

2.  **Database Scaling:** If we actually tried to load 1 million songs into our local Python dictionary, it would generate trillions of hash pairs and completely crash our computer\'s RAM. Furthermore, moments of absolute silence in songs produce identical \"zero\" hashes, which would create massive traffic jams on a single matching server. To scale this up to 50 million songs, we would have to move away from local RAM and use advanced NoSQL databases distributed across multiple cloud servers.

**6. GitHub Repository and Video Explanation:**

The complete source code (Python), DSP evaluation matrices, Unit testing harnesses, and demo artifacts for this codebase are hosted publicly.

Access Repository Here:\
[**https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-**](https://github.com/itsrahuldah/AUDIO-FINGERNPRINTING-)

The video explanation of the Shazam algorithm can be accessed on youtube here:

**<https://youtu.be/BTwkqNeb3HA?si=WrsdnJ20B06Am7ef>**

**7. References:**

1.  Wang, A. L.-C. \"An Industrial-Strength Audio Search Algorithm\". Proceedings of the 4th International Society for Music Information Retrieval Conference (ISMIR), Shazam Entertainment, Ltd. 2003.

> <https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf>

2.  Haitsma, J. and Kalker, A., \"A Highly Robust Audio Fingerprinting System\", ISMIR 2002, pp. 107-115.

3.  Yang, C., \"MACS: Music Audio Characteristic Sequence Indexing For Similarity Retrieval\", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics, 2001.
