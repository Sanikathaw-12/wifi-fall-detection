# WiFi-Based Fall Detection using CSI

A machine learning system that detects human falls using WiFi Channel State
Information (CSI) — no wearable device or camera required. Built as a
camera-free, privacy-preserving alternative to vision-based elderly-care
monitoring systems.

## How it works
WiFi signals bounce off people in a room (multipath), and these reflections
are captured as CSI — amplitude/phase values across subcarriers and antennas.
A fall causes a distinct pattern: a sharp, brief disturbance in the signal
followed by stillness, unlike walking (a sustained, periodic pattern) or
sitting (gradual change).

## Dataset
[UT-HAR dataset](https://github.com/ermongroup/Wifi_Activity_Recognition)
(Yousefi et al., IEEE Communications Magazine 2017) — 7 activities: lie down,
fall, walk, pickup, run, sit down, stand up. ~4,973 samples total, each a
250-timestep × 90-value (30 subcarriers × 3 antennas) CSI window.

Download the preprocessed data from [this Google Drive folder]
(https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt)
(UT_HAR subfolder only) and place it at `data/UT_HAR/`.

## Approach
- Binary classification: fall vs. all other activities
- Features: mean, standard deviation, and max amplitude change per CSI
  subcarrier across the time window
- Model: Random Forest (class-balanced, to handle fall being a minority class)

## Results
| Model | Accuracy | Precision (fall) | Recall (fall) |
|---|---|---|---|
| Random Forest (class-balanced) | 97.8% | 0.95 | 0.80 |
| LSTM (class-weighted, best checkpoint) | 92.8% | 0.57 | 0.87 |

**Note:** recall matters more than accuracy here — a missed fall is the
costly failure mode for a real safety system. The dataset is imbalanced
(only ~9% of samples are falls). The LSTM catches more falls (87% vs 80%)
at the cost of more false alarms (precision drops from 0.95 to 0.57) —
a genuine trade-off. For a real deployment, recall would likely be
prioritized, making the LSTM the stronger candidate despite lower
raw accuracy.

## Setup
```
pip install -r requirements.txt
python src/train_baseline.py
```


## Acknowledgments
Data loading approach adapted from [SenseFi]
(https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) (Yang et al., 2023).

## Future work
- Deep learning models (LSTM/CNN) on raw CSI instead of handcrafted features
- Real hardware capture using ESP32 CSI Tool for live testing
- Multimodal fusion with accelerometer-based fall detection