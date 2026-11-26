# NEURO//WAVE 🌊🧠

**Teaching a computer to listen for ripples in space.**

> 🚧 **Status: work in progress.** This is a learning project. The AI detectors in it do **not** work well yet, and this README tells you honestly why. See [Known mistakes](#-known-mistakes-and-problems-in-the-notebook) at the end.

---

## 📖 Contents

1. [The big idea (super simple)](#-the-big-idea-super-simple)
2. [What is a gravitational wave?](#-what-is-a-gravitational-wave)
3. [The two events I used](#-the-two-events-i-used)
4. [What I did, step by step](#-what-i-did-step-by-step)
5. [Results (the honest version)](#-results-the-honest-version)
6. [Plots the notebook saves](#-plots-the-notebook-saves)
7. [Tools and data credits](#-tools-and-data-credits)
8. [Limitations](#-limitations-what-this-project-is-not)
9. [Future work](#-future-work)
10. [Known mistakes and problems in the notebook](#-known-mistakes-and-problems-in-the-notebook)
11. [Ongoing work](#-ongoing-work)

---

## 🌈 The big idea (super simple)

Imagine you drop a stone into a pond. Ripples spread out in rings.

Space does the same thing! When two very heavy things (like black holes) crash into each other, they make **ripples in space itself**. These ripples are called **gravitational waves**.

The ripples are *super* tiny by the time they reach Earth. Scientists built giant machines called **LIGO** to feel them. But the machines also hear lots of noise, a bit like trying to hear a whisper in a loud classroom.

**My project asks:** *can a computer (an AI) learn to spot the whisper inside the noise?*

---

## 🌌 What is a gravitational wave?

- Space is not empty. Think of it like a stretchy **trampoline sheet**. Heavy things make dents in it.
- When two heavy things **spin around each other and crash**, they shake the sheet. The shaking travels outward as waves at the speed of light.
- When the wave passes Earth, it **stretches and squeezes space by a tiny, tiny amount**: much smaller than a proton. The size of that stretch is called **strain**. For these events it is around `1e-21` (a 1 with 21 zeros after the decimal point!).
- The signal sounds like a **bird chirp that goes higher and higher** and then stops. Scientists call it a **chirp**. It gets higher in pitch because the two objects spin faster and faster as they get closer.

### How LIGO listens

LIGO has two detectors that are far apart:

| Name | Where | Short name |
|------|-------|-----------|
| LIGO Hanford | Washington, USA | **H1** |
| LIGO Livingston | Louisiana, USA | **L1** |

Each has two 4-kilometre-long arms shaped like an "L". A real wave makes the arms change length by a tiny amount. Because the two detectors are about 3,000 km apart, a wave reaches one a few milliseconds before the other (at most about 10 ms). Seeing the same chirp in **both** is a big clue that it is real and not just local noise.

---

## ⭐ The two events I used

| Event | Date | What crashed | Short label in code |
|-------|------|--------------|--------------------|
| **GW150914** | 14 Sept 2015 | Two **black holes** (about 36 and 29 times the Sun's mass) | `BBH` (binary black hole) |
| **GW170817** | 17 Aug 2017 | Two **neutron stars** (super-dense dead stars) | `BNS` (binary neutron star) |

GW150914 was the **first** gravitational wave ever detected by humans. GW170817 was special because telescopes also saw light from it.

---

## 🛠️ What I did, step by step

Everything lives in one Google Colab notebook: **`NeuroGW.ipynb`**.

### Step 1: Get real data 📥
I downloaded **real LIGO data** (free and open) for both events from both detectors (H1 and L1).
- 4 files in total (2 events x 2 detectors).
- Each file is **32 seconds** long at **16,384 samples per second** (524,288 numbers per file).
- All 4 passed my checks (no missing or broken numbers).

### Step 2: Look at the data 🔍
- **Raw strain plots:** just the wiggly line over time.
- **PSD (Power Spectral Density):** a chart showing *how loud the noise is at each pitch*. Detector noise is very loud at some pitches and quiet at others.
- **Whitening:** turn down the loud pitches and turn up the quiet ones, so the noise is "flat" and a hidden chirp can stand out.
- **Band-pass filter (20-1000 Hz):** keep only the pitches where black-hole chirps live.
- **Spectrograms:** pictures showing pitch over time (like a music visualiser).
- **H1 vs L1 comparison:** lay the two detectors on top of each other.

### Step 3: Collect "just noise" (background) 🔇
For an AI to learn what noise looks like, I downloaded **real detector noise** from times when no known event was happening (before each event).
- I asked for 10 stretches of 256 seconds per event, using 4,096 samples per second.
- **15 of 20 worked.** For GW150914, 5 worked and 5 failed (no usable data for those times). For GW170817, all 10 worked.
- I cut them into **480 windows of 8 seconds** each. Each window has 2 channels (H1 and L1).

### Step 4: Clean the noise 🧼
For each stretch I measured the noise PSD, whitened it, kept only 20-500 Hz, and shrank it from 4,096 to **1,024 samples per second**. Each window became a table of **8,192 numbers x 2 detectors**.

### Step 5: Make fake chirps (simulation) 🧪
I used **PyCBC** to simulate black-hole mergers (`IMRPhenomD` model, 400 Mpc distance). I made **8 pairs of black-hole masses**:

`(20,15)  (25,20)  (30,20)  (36,29)  (40,30)  (45,35)  (50,40)  (55,45)`

(units: times the mass of the Sun.)

### Step 6: Hide the fake chirps inside real noise 🕵️
This is called **injection**: take a real noise window and add a simulated chirp, giving the AI "noise" examples and "noise + chirp" examples to learn from. I added a random mix of the two wave shapes (polarisations), a random time difference between H1 and L1 (up to about +/- 10 ms), a random size for L1, and a random position in the window.

I split the data **by time interval** (not at random) so that the AI is never tested on noise it has already seen.

### Step 7: Train AIs 🤖
I tried three kinds of AI:

| # | Model | Idea |
|---|-------|------|
| 1 | **Simple 1D CNN** | A basic pattern-finder for waves |
| 2 | **Physics-informed CNN + BiGRU** | First layer starts out as copies of the chirp shapes (like a *matched filter*), then a "memory" layer (BiGRU) reads the pattern over time |
| 3 | **Convolutional autoencoder** (anomaly detector) | Trained on **noise only**. It learns to rebuild noise. If it rebuilds a window badly, that window is "weird" and might hold a signal |

### Step 8: Test the autoencoder with fake signals 🎯
I added fake chirps of 5 different strengths, for all 8 mass pairs, to the 96 held-out noise windows (3,840 tests) and checked how many the autoencoder flagged.

---

## 📊 Results (the honest version)

### Short summary
**The classifier AIs did not learn to find the signal (they scored about as well as a coin flip), and the autoencoder test was too easy to prove anything.** The reasons are in the mistakes section.

### Model 1: Simple CNN
| Thing | Value |
|-------|-------|
| Training examples | 267 (validation 57, test 60) |
| Validation accuracy | 0.667 |
| Validation AUC | 0.500 |

Accuracy 0.667 looks okay, but it is a trick: two-thirds of the examples were "signal", so the model just said "signal" every time. **AUC 0.5 = a coin flip.**

### Model 2: Physics-informed CNN + BiGRU
| Version | Data | ROC-AUC | Accuracy |
|---------|------|---------|----------|
| First try | 2,400 train / 600 val | 0.519 | 0.50 |
| **Final** | 4,000 train / 1,000 val | **0.552** | **0.504** |

- Average precision (final): 0.566
- Confusion matrix (final): `[[71, 429], [67, 433]]`. It said "signal" for almost everything: it caught 87% of signals but also called 86% of pure-noise windows "signal".
- 0.5 means guessing and 1.0 means perfect, so **0.552 is barely better than guessing.**

### Model 3: Autoencoder (anomaly detector)
| Thing | Value |
|-------|-------|
| Windows: train / calibration / test | 288 / 96 / 96 (from 9 / 3 / 3 time intervals) |
| Parameters | 376,962 |
| Best validation error (MAE) | 0.0883 (epoch 38) |
| Threshold (95th percentile of calibration) | 0.1767 |
| Calibration error: mean / max | 0.0883 / 0.2412 |
| Test error: mean / max | 0.0647 / 0.0923 |
| Clean test windows flagged | **0 of 96** |

### Autoencoder injection test
| "Strength" | Windows tested | Flagged |
|-----------|----------------|---------|
| 5, 8, 12, 16, 20 | 768 each (8 masses x 96 windows) | **100% at every strength** |

⚠️ **Do not read this as "the detector works perfectly."** The "strength" number is **not** a real signal-to-noise ratio, and the added signals were enormous. See [mistake #2](#-known-mistakes-and-problems-in-the-notebook).

---

## 🧰 Tools and data credits

- **Data:** open data from the [Gravitational Wave Open Science Center (GWOSC)](https://gwosc.org), provided by LIGO, Virgo and KAGRA. *This research has made use of data obtained from the Gravitational Wave Open Science Center.*
- **Python libraries:** `gwpy`, `gwosc`, `pycbc`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow` / `keras` (TensorFlow 2.21.0 in my run)
- **Runs in:** Google Colab with Google Drive for storage

---

## ⚠️ Limitations (what this project is NOT)

- ❌ It is **not** a working gravitational-wave detector.
- ❌ It has **not** found any new events, and I make no discovery claims.
- ❌ The final models were **never run on the real GW150914 / GW170817 signals**. Only simulated injections were used.
- ❌ The "SNR" numbers are **proxies** (my own size measures), not real matched-filter SNR.
- ❌ Only **black-hole mergers** were simulated in the final dataset. Neutron-star (BNS) waveforms were used in the very first dataset only.
- ❌ The detector response is **simplified**: no real antenna patterns, sky position or detector orientation.
- ❌ Small data: 15 noise intervals and 480 windows, many re-used (sampled with replacement) to make thousands of examples.
- ❌ There is no real **matched-filter baseline** to compare against yet.
- ❌ Real LIGO noise has glitches and changes over time. I did not handle glitches.

---

## 🔭 Future work

1. **Fix the injections** (check the chirp is really inside the injected window) and plot examples before training.
2. **Use real SNR:** compute proper matched-filter SNR with PyCBC and inject at SNR values like 8, 10, 15, 20.
3. **Run a real matched-filter baseline** and compare the AI against it (ROC curve, and detection rate at a fixed false-alarm rate).
4. **Proper detector response:** use real antenna patterns, random sky positions and the real H1-L1 time delay.
5. **Better data handling:** use float64 (or rescale strain) to avoid number underflow, taper windows before the FFT, and whiten the standard way.
6. **Test on the real events** (GW150914, GW170817) and on known glitches.
7. **Add neutron-star (BNS) signals** and more masses and spins.
8. **Strengthen the autoencoder:** a tighter bottleneck, more noise data, and a threshold chosen from far more windows.
9. **Report honest metrics:** false-alarm rate, detection efficiency vs SNR, and several random seeds.
10. **Clean the notebook** into a reusable Python package with tests.

---

## 🐛 Known mistakes and problems in the notebook

I re-read my own notebook and listed what I found. Writing them down is part of learning. Most important first.

1. **The injected chirps in the final dataset are probably missing the merger.** `WAVEFORM_BANK` waveforms peak around `1e-21`, but `WAVEFORMS_FOR_INJECTION` (the last 4 seconds only) peak around `1e-26` to `1e-25`, roughly 9,000 to 60,000 times smaller. The loudest part of the chirp is very likely **not** in that slice, so the AI was being asked to find a faint tail that was then rescaled to look "loud". This alone can explain why AUC stayed near 0.5. The template filters inside the CNN were also cut from the last 1 second of the waveform, so they may have the same problem.
2. **The autoencoder "SNR" test is not a real test.** The scale used is `target x (size of background) / (size of template)`, so at "strength 5" the added signal is **5 times bigger than the whole noise window**, and at 20 it is 20 times bigger. Of course 100% were flagged. It only shows that giant additions look weird, not that real signals are detectable.
3. **The matched-filter baseline was never run.** The section is titled "matched-filter baseline" but only generates one reference waveform and stops. `matched_filter` is imported but not used.
4. **Tiny-number problems.** Strain is around `1e-19` to `1e-21` and was stored as `float32`. The saved PSD values reach `0.0` at high frequencies, so the whitening used an artificial floor. Whitened windows range about +/-3 million rather than being near unit size, and I did not taper the windows before the FFT (edge effects). The first CNN was also trained on raw `~1e-19` numbers, which is hard for a network to learn from.
5. **Plus/cross used as detector channels.** In the autoencoder benchmark the two wave polarisations were put into the two channels as if they were H1/L1. They are not. (I wrote this limitation in the methodology file already.)
6. **Template filters are too coarse.** The first layer moves 8 samples at a time (about 7.8 ms), but chirp wiggles at 100+ Hz repeat faster than that, so it can miss the best alignment.
7. **The autoencoder bottleneck is weak.** It squeezes 16,384 numbers into 8,192 (only 2x), so it may be able to rebuild almost anything. Also, the threshold came from only 3 calibration intervals that included at least one high-error window (max 0.241), while the test windows were calmer (max 0.092), so "0 of 96 flagged" is not a reliable false-alarm rate.
8. **Hard-coded baseline number.** I typed `0.51524` as the "previous AUC", but the notebook actually printed `0.51897`. The "improvement of +0.037" is therefore not computed correctly, and with only 1,000 validation samples it is small anyway.
9. **Wrong history plot.** `04_training_history.png` is built from the variable `history`, which still held the earlier CNN-BiGRU training, but the title says "Autoencoder". The autoencoder's history is `history_ae`. Please regenerate.
10. **First autoencoder scoring used un-normalised data.** The first run gave errors around `1e7` because the model expects standardised input. I corrected it later ("CORRECTED FINAL RESULTS"), but earlier outputs with the same file names may have been overwritten or mixed up. Use only the corrected results.
11. **Naming and scope confusion.** Files are called `GW150914_...` but the background data comes from both GW150914 and GW170817 times. The project is written as `NEURO_WAVE` in folder names and `NEURO//WAVE` in printed messages, so I should pick one name.
12. **Repeated examples.** The 4,000 and 1,000 examples are drawn with replacement from only a few hundred windows, so the same noise appears many times. The split is by time interval (good), but the effective data size is much smaller than it looks.
13. **Cell-order problems.** The first download cell fails (`download_event_data` not defined) until the fixed acquisition cell runs. `pycbc` is imported before it is installed, and `TimeSeries` is imported from both `gwpy` and `pycbc` (the later import wins, which can break `fetch_open_data`). Two Drive mount points (`/content/drive` and `/content/gdrive`) are also used. On a fresh runtime, reorder the cells, install everything at the top, and use one path.

---

## 🌱 Ongoing work

This is **an ongoing project**. I am still learning physics and computer science, and I will keep updating the notebook and this README as my understanding gets better. I expect to fix the problems above, add proper matched-filter comparisons, and test on real events.

If you spot something wrong, I would love to hear it. Being wrong and finding out why is how this project teaches me. 💫

*Author: Amna.*
