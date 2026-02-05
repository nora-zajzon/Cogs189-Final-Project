"""
Train motor imagery classifier: CSP + LDA.
Loads saved EEG + events from run_vep.py (calibration mode), epochs, preprocesses,
extracts CSP features, trains LDA, reports accuracy, saves model and CSP.
Same process as original: run this script after collecting data with run_vep.py.
"""
import os
import argparse
import numpy as np
import pickle
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mne.decoding import CSP

# --- Params ---
sampling_rate = 250
epoch_tstart = 0.5   # s after cue
epoch_tend = 3.5     # s after cue → 3 s of data
bandpass_low, bandpass_high = 8, 30
notch_freq = 60.0
n_csp_components = 4
data_dir = "data"
model_cache_dir = "cache"
dataset_name = "motor_imagery_2class"


def get_save_dir(subject, session):
    return os.path.join(data_dir, dataset_name, f"sub-{subject:02d}", f"ses-{session:02d}")


def load_events(save_dir, run=1):
    path = os.path.join(save_dir, f"events_run-{run}.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True)
    events = arr.item() if arr.ndim == 0 else arr.tolist()
    if not events:
        return []
    out = []
    for ev in events:
        if isinstance(ev, dict):
            out.append({"sample": int(ev["sample"]), "label": int(ev["label"])})
        else:
            out.append({"sample": int(ev[0]), "label": int(ev[1])})
    return out


def load_run(save_dir, run=1):
    eeg_path = os.path.join(save_dir, f"eeg_run-{run}.npy")
    if not os.path.exists(eeg_path):
        return None, None
    eeg = np.load(eeg_path)
    events = load_events(save_dir, run=run)
    return eeg, events if events is not None else []


def load_all_data(subject=1, session=1, runs=None):
    save_dir = get_save_dir(subject, session)
    if runs is None:
        runs = [1]
    all_events = []
    all_eeg = []
    sample_offset = 0
    for r in runs:
        eeg, events = load_run(save_dir, run=r)
        if eeg is None or not events:
            continue
        if all_eeg:
            for ev in events:
                all_events.append({"sample": ev["sample"] + sample_offset, "label": ev["label"]})
            all_eeg.append(eeg)
            sample_offset += eeg.shape[1]
        else:
            all_events = [dict(ev) for ev in events]
            all_eeg = [eeg]
            sample_offset = eeg.shape[1]
    if not all_eeg:
        return None, None
    return np.hstack(all_eeg), all_events


def events_to_epochs(continuous_eeg, events, sfreq, t_start=0.5, t_end=3.5):
    n_channels, n_total = continuous_eeg.shape
    n_samples_epoch = int((t_end - t_start) * sfreq)
    epoch_start_offset = int(t_start * sfreq)
    epochs_list, labels_list = [], []
    for ev in events:
        s0 = ev["sample"] + epoch_start_offset
        s1 = s0 + n_samples_epoch
        if s0 < 0 or s1 > n_total:
            continue
        epochs_list.append(continuous_eeg[:, s0:s1].copy())
        labels_list.append(ev["label"])
    if not epochs_list:
        return np.zeros((0, n_channels, n_samples_epoch)), np.array([], dtype=int)
    return np.stack(epochs_list, axis=0), np.array(labels_list, dtype=int)


def preprocess_epochs(epochs, sfreq):
    """Bandpass 8-30 Hz, notch 60 Hz, baseline subtract."""
    out = np.zeros_like(epochs, dtype=np.float64)
    for i in range(epochs.shape[0]):
        x = mne.filter.notch_filter(epochs[i], Fs=sfreq, freqs=notch_freq, verbose=False)
        x = mne.filter.filter_data(x, sfreq=sfreq, l_freq=bandpass_low, h_freq=bandpass_high, verbose=False)
        n_baseline = min(int(0.5 * sfreq), x.shape[1] // 5)
        if n_baseline > 0:
            x = x - np.mean(x[:, :n_baseline], axis=1, keepdims=True)
        out[i] = x
    return out


def train_pipeline(subject=1, session=1, runs=None, test_size=0.25, random_state=42):
    eeg, events = load_all_data(subject=subject, session=session, runs=runs)
    if eeg is None or not events:
        raise FileNotFoundError(
            f"No data found for sub-{subject:02d} ses-{session:02d}. "
            "Run run_vep.py with calibration_mode=True first."
        )

    epochs, labels = events_to_epochs(eeg, events, sampling_rate, t_start=epoch_tstart, t_end=epoch_tend)
    if len(labels) == 0:
        raise ValueError("No valid epochs extracted.")

    epochs = preprocess_epochs(epochs, sampling_rate)

    X_train, X_test, y_train, y_test = train_test_split(
        epochs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    n_components = min(n_csp_components, X_train.shape[1], X_train.shape[2] - 1)
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat = csp.transform(X_test)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train_feat, y_train)
    y_pred = model.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test accuracy: {acc:.2%}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    os.makedirs(model_cache_dir, exist_ok=True)
    with open(os.path.join(model_cache_dir, "motor_lda_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_cache_dir, "motor_csp.pkl"), "wb") as f:
        pickle.dump(csp, f)

    return acc, csp, model


def main():
    parser = argparse.ArgumentParser(description="Train motor imagery CSP+LDA classifier")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--runs", type=str, default="1", help="Comma-separated run numbers")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = [int(x) for x in args.runs.split(",")]
    train_pipeline(
        subject=args.subject, session=args.session, runs=runs,
        test_size=args.test_size, random_state=args.seed,
    )


if __name__ == "__main__":
    main()
