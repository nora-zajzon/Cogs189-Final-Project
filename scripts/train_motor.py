import numpy as np
import os
import pickle
import argparse

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

parser = argparse.ArgumentParser(description='Train motor imagery (CSP+LDA) model')

folder_path = '../data/motor_imagery_2class/sub-01/ses-04/'
model_save_dir = 'cache/'
model_name = 'motor_lda_model.pkl'
csp_name = 'motor_csp.pkl'

sampling_rate = 250

baseline_duration = 0.2 # first 0.2 seconds of each trial are considered baseline
baseline_samples = int(baseline_duration * sampling_rate) # 50 samples

run_files = [ # just getting raw eeg signals for ecah trial
    f for f in os.listdir(folder_path)
    if f.startswith('eeg-trials_') and f.endswith('.npy')
]

if len(run_files) == 0:
    raise FileNotFoundError("No motor imagery trials found. Run run_mi.py calibration first.")


all_epochs = []
all_labels = []

for run_file in run_files:

    run_number = int(run_file.split('-')[-1].split('.')[0])

    eeg_trials = np.load(os.path.join(folder_path, run_file)) # loads EEG data  
    events = np.load( # loads event data
        os.path.join(folder_path, f'events_{run_file.split("eeg-trials_")[1]}'),
        allow_pickle=True # allows to load dictionary with label
    )

    # eeg_trials shape:
    # (n_trials, n_channels, samples)

    labels = np.array([ev["label"] for ev in events], dtype=int) # array of labels

    if len(labels) != len(eeg_trials):
        raise ValueError("Mismatch between trials and labels")

    all_epochs.append(eeg_trials)
    all_labels.append(labels)
    
# Combine all trials into one dataset
epochs = np.concatenate(all_epochs, axis=0)
labels = np.concatenate(all_labels, axis=0)

print("Combined shape:", epochs.shape, "labels:", labels.shape)

# Baseline cropped out (isolate motor imagery)
epochs = epochs[:, :, baseline_samples:]

# Training (CSP+LDA)
def run_motor_csp_lda(epochs, labels, test_size=0.25, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        epochs, # X
        labels, # y
        test_size=test_size,
        stratify=labels, # balances classes
        random_state=random_state
    )

    # mean subtract for each trial in each channel to make sure signal is centered around 0
    X_train = X_train - np.mean(X_train, axis=-1, keepdims=True)
    X_test = X_test - np.mean(X_test, axis=-1, keepdims=True)

    # X_train.shape[1] = number of channels
    # X_train.shape[2] = number of time samples
    # for 2-class CSP, a small number of components is common
    n_components = min(4, X_train.shape[1], X_train.shape[2] - 1)

    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False
    )

    X_train_feat = csp.fit_transform(X_train, y_train) # learn the spatial filters using X_train and y_train
    X_test_feat = csp.transform(X_test) # convert training EEG into CSP filters
    # turns into (n_trials, 4)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train_feat, y_train)
    # ex: high feature 1 + low feature 4 -> left-hand imagery
    # ex: low feature 1 + high feature 4 -> right-hand imagery

    preds = model.predict(X_test_feat)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, normalize='true')

    # Ex:
    # pred class 0, pred class 1
    # 0.83 0.17 true class 0
    # 0.20 0.80 true class 1

    print(f"Model: CSP+LDA  Acc: {acc:.3f}")

    return cm, acc, model, csp

cm, acc, model, csp = run_motor_csp_lda(epochs, labels)

os.makedirs(model_save_dir, exist_ok=True)

with open(os.path.join(model_save_dir, model_name), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(model_save_dir, csp_name), 'wb') as f:
    pickle.dump(csp, f)

print("Saved model + CSP to cache/")