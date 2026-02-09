import numpy as np
import os
import pickle
import argparse

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

parser = argparse.ArgumentParser(description='Train motor imagery (CSP+LDA) model')

folder_path = 'data/motor_imagery_2class/sub-01/ses-01/'
model_save_dir = 'cache/'
model_name = 'motor_lda_model.pkl'
csp_name = 'motor_csp.pkl'

sampling_rate = 250

baseline_duration = 0.2
baseline_samples = int(baseline_duration * sampling_rate)

run_files = [
    f for f in os.listdir(folder_path)
    if f.startswith('eeg-trials_') and f.endswith('.npy')
]

if len(run_files) == 0:
    raise FileNotFoundError("No motor imagery trials found. Run run_mi.py calibration first.")


all_epochs = []
all_labels = []

for run_file in run_files:

    run_number = int(run_file.split('-')[-1].split('.')[0])

    eeg_trials = np.load(os.path.join(folder_path, run_file))
    events = np.load(
        os.path.join(folder_path, f'events_{run_file.split("eeg-trials_")[1]}'),
        allow_pickle=True
    )

    # eeg_trials shape:
    # (n_trials, n_channels, samples)

    labels = np.array([ev["label"] for ev in events], dtype=int)

    if len(labels) != len(eeg_trials):
        raise ValueError("Mismatch between trials and labels")

    all_epochs.append(eeg_trials)
    all_labels.append(labels)
    
# Combine runs
epochs = np.concatenate(all_epochs, axis=0)
labels = np.concatenate(all_labels, axis=0)

print("Combined shape:", epochs.shape, "labels:", labels.shape)

# Baseline crop (same philosophy as train_trca)
epochs = epochs[:, :, baseline_samples:]

# Training (same idea as TRCA pipeline but CSP+LDA)
def run_motor_csp_lda(epochs, labels, test_size=0.25, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        epochs,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # mean subtract (same step as TRCA)
    X_train = X_train - np.mean(X_train, axis=-1, keepdims=True)
    X_test = X_test - np.mean(X_test, axis=-1, keepdims=True)

    n_components = min(4, X_train.shape[1], X_train.shape[2] - 1)

    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat = csp.transform(X_test)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train_feat, y_train)

    preds = model.predict(X_test_feat)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, normalize='true')

    print(f"Model: CSP+LDA  Acc: {acc:.3f}")

    return cm, acc, model, csp


cm, acc, model, csp = run_motor_csp_lda(epochs, labels)

os.makedirs(model_save_dir, exist_ok=True)

with open(os.path.join(model_save_dir, model_name), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(model_save_dir, csp_name), 'wb') as f:
    pickle.dump(csp, f)

print("Saved model + CSP to cache/")
