# Motor Imagery EEG Snake Game

Control a snake game using motor imagery EEG signals. Imagine squeezing your non-dominant hand to move LEFT, or imagine squeezing your dominant foot to move RIGHT.

## Setup for Windows 11

```
pip install virtualenv
virtualenv pyenv --python=3.11.9
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pyenv\Scripts\activate
```

Install "Desktop development with C++" workload: https://visualstudio.microsoft.com/visual-cpp-build-tools/

```
pip install -r requirements.txt
git clone https://github.com/TBC-TJU/brainda.git
cd brainda
pip install -r requirements.txt
pip install -e .
```

All Python packages are installed into `pyenv/` and can be removed by deleting that folder. Activate the environment with `pyenv\Scripts\activate`, deactivate with `deactivate`. You may need administrator access for `pip install -r requirements.txt`.

## Setup for macOS

```
pip install virtualenv
virtualenv pyenv --python=3.11.9
source pyenv/bin/activate
pip install -r requirements.txt
git clone https://github.com/TBC-TJU/brainda.git
cd brainda
pip install -r requirements.txt
pip install -e .
```

## How It Works

There are three phases:

1. **Calibration** - Collect labeled EEG data while the user imagines motor movements
2. **Training** - Train a CSP+LDA classifier on the collected data
3. **Snake Game** - Play the snake game controlled by real-time EEG classification

### Phase 1: Calibration (Data Collection)

In `run_mi.py`, set:

```python
calibration_mode = False
```

Then run:

```
python run_mi.py
```

The program will guide you through trials with a countdown:
- "Prepare: Move LEFT HAND" (or "Move RIGHT FOOT")
- 3... 2... 1...
- RECORDING

During RECORDING, imagine squeezing your non-dominant hand (class 0) or dominant foot (class 1) as instructed. EEG data is saved to `data/motor_imagery_2class/sub-01/ses-01/`.

Press ESC at any time to save data and exit early.

### Phase 2: Training

```
python scripts/train_motor.py
```

This trains a CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis) classifier on the collected calibration data. The trained model is saved to `cache/`.

### Phase 3: Snake Game (Real-time BCI Control)

In `run_mi.py`, set:

```python
calibration_mode = True
```

Then run:

```
python run_mi.py
```

The snake game will start. Your EEG signals control the snake:
- **Imagine non-dominant hand** (class 0) -> snake turns LEFT
- **Imagine dominant foot** (class 1) -> snake turns RIGHT

The snake moves forward automatically. Your brain signals steer it left or right. Eat the red food to score points. The game ends if the snake collides with itself.

Press ESC to quit.

## Improving Accuracy

Repeat calibration and training with more data for better accuracy:

1. Run `python run_mi.py` with `calibration_mode = False` (change `run = 1` to `run = 2`, etc. for additional runs)
2. Run `python scripts/train_motor.py`
3. Test with `calibration_mode = True`

More calibration runs generally improve classification accuracy.

## Parameters (in run_mi.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calibration_mode` | `False` | `False` = collect data, `True` = play snake game |
| `n_per_class` | `30` | Trials per class during calibration |
| `run` | `1` | Run number (use as seed, increment for additional runs) |
| `subject` | `1` | Subject ID |
| `session` | `1` | Session ID |
| `baseline_duration` | `1.5` | Seconds of baseline before recording |
| `record_duration` | `3.0` | Seconds of motor imagery recording per trial |
| `realtime_window_sec` | `1.5` | EEG window size for real-time classification |
| `smoothing_n_predictions` | `5` | Number of predictions to average for smoothing |

## File Structure

```
run_mi.py                  # Main script (calibration + snake game)
scripts/train_motor.py     # Model training script
cache/                     # Trained model files (auto-created)
  motor_lda_model.pkl
  motor_csp.pkl
data/motor_imagery_2class/ # Collected EEG data (auto-created)
  sub-01/ses-01/
    eeg-trials_*.npy
    events_*.npy
    ...
```

Note: The `snakegame/` directory contains a standalone JavaScript browser game and is NOT used by the BCI pipeline. The snake game is built into `run_mi.py` using PsychoPy.
