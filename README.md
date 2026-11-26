# NEURO//WAVE 🌊🧠

**Teaching a computer to listen for ripples in space.**

> 🚧 **Status: work in progress.** This is a learning project. The AI detectors in it do **not** work well yet, and this README tells you honestly why. See [Lessons learned](#-lessons-learned-and-what-i-want-to-improve) near the end.

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
10. [Lessons learned and what I want to improve](#-lessons-learned-and-what-i-want-to-improve)
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
**The classifier AIs did not learn to find the signal (they scored about as well as a coin flip), and the autoencoder test was too easy to prove anything.** The reasons are in the lessons-learned section.

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

⚠️ **Do not read this as "the detector works perfectly."** The "strength" number is **not** a real signal-to-noise ratio, and the added signals were enormous. See [Lessons learned](#-lessons-learned-and-what-i-want-to-improve).

---

## 🖼️ Plots the notebook saves

| File | What it shows |
|------|---------------|
| `raw_strain.png` | Raw strain for each event and detector |
| `01_error_distribution.png` | Autoencoder error: calibration vs test windows |
| `02_anomaly_timeline.png` | Error for each test window vs threshold |
| `03_channel_comparison.png` | H1-channel error vs L1-channel error |
| `04_training_history.png` | ⚠️ Probably the **wrong** curve, see [Lessons learned](#-lessons-learned-and-what-i-want-to-improve) |
| `05_morphology_injection_response.png` | % flagged vs injection "strength" |
| `06_injection_error_distributions.png` | Error with and without injected signals |
| `07_mass_configuration_comparison.png` | Flagging for each of the 8 mass pairs |

Data files, models and results are saved to Google Drive under `NEURO_WAVE/` (`data/raw`, `data/background`, `data/ml_dataset`, `data/anomaly_detection`, `results/final_anomaly_detection`). They are not stored in this repo because they are large.

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

## 🌟 Lessons learned and what I want to improve

Every real research project has bugs and rough edges, and finding them is part of the job. Scientists call this *debugging*. I re-read my notebook and grouped what I learned into a few themes. Each one is a clear next step, not a failure.

| Theme | What I learned | Next step |
|-------|----------------|-----------|
| 🧪 **Practice signals** | I need to double-check that the fake chirps hidden in the noise really contain the loudest part (the merger). One of my checks suggests the final dataset may have trimmed it off. This is probably the main reason the classifier stayed near coin-flip level. | Plot a few injected examples *before* training, and keep the full chirp. |
| 📏 **Measuring signal strength** | I used a simple size measure instead of real signal-to-noise ratio (SNR). In the autoencoder test, that made the added signals very large, so "100% flagged" was too easy to be meaningful. | Use proper matched-filter SNR and test at realistic strengths. |
| 🏁 **Comparing with a standard method** | Good AI results should be compared with the classic **matched filter**. I set up the reference waveform but have not run the comparison yet. | Run the baseline and compare ROC curves. |
| 🔢 **Very tiny numbers** | Strain values are around `1e-21`, which computers can struggle with. Storing them in a small number format and not smoothing the window edges before whitening may have caused small errors. | Rescale or use higher precision, taper windows, and re-check the whitening. |
| 🎛️ **Simplified physics** | In places I used simple stand-ins (for example, putting the two wave polarisations into the two channels instead of real H1 and L1 detector responses, and a fairly coarse template filter step). These are fine for a first prototype. | Add real detector response and finer filtering. |
| 📦 **Small data and thresholds** | I had a small amount of noise data (15 intervals), re-used many times. The autoencoder compresses the input only a little, and its alarm threshold came from just 3 intervals, so "0 false alarms" is not yet a reliable number. | Gather more noise, tighten the autoencoder, and set the threshold from more data. |
| 🧹 **Notebook housekeeping** | A few things need tidying: one plot (`04_training_history.png`) was drawn from the wrong training history, one comparison number was typed by hand, my first autoencoder scoring used un-normalised input (fixed later in the "corrected" cells), some cells need to run in a particular order, and file names mix `GW150914` and `GW170817` data. | Reorder cells, regenerate plots, compute every number in code, and rename files. |

**The honest takeaway:** the detectors in this notebook are early prototypes and do not detect signals reliably yet. Reporting that clearly, and knowing exactly what to fix, is good science. 🚀

---

## 🌱 Ongoing work

This is **an ongoing project**. I am still learning physics and computer science, and I will keep updating the notebook and this README as my understanding gets better. I expect to fix the problems above, add proper matched-filter comparisons, and test on real events.

If you spot something wrong, I would love to hear it. Being wrong and finding out why is how this project teaches me. 💫

*Author: Amna.*
