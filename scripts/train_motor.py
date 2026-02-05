import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Train motor imagery (CSP+LDA) model')

folder_path = 'data/motor_imagery_2class/sub-01/ses-01/'
model_save_dir = 'cache/'
model_name = 'motor_lda_model.pkl'
csp_name = 'motor_csp.pkl'
sampling_rate = 250

# Epoch window: 0.5 s after cue -> 3.5 s (3 s of data)
epoch_tstart = 0.5
epoch_tend = 3.5
baseline_duration = 0.2
baseline_samples = int(baseline_duration * sampling_rate)

# List all run files in the folder (eeg_run-X.npy)
if not os.path.isdir(folder_path):
    raise FileNotFoundError(f"No data folder: {folder_path}. Run run_vep.py with calibration_mode=True first.")
run_files = [f for f in os.listdir(folder_path) if f.startswith('eeg_run-') and f.endswith('.npy')]

epochs_list = []
labels_list = []

for run_file in run_files:
    run_number = int(run_file.split('-')[-1].split('.')[0])
    eeg = np.load(os.path.join(folder_path, f'eeg_run-{run_number}.npy'))
    events_arr = np.load(os.path.join(folder_path, f'events_run-{run_number}.npy'), allow_pickle=True)
    events = events_arr.item() if events_arr.ndim == 0 else events_arr.tolist()
    if not events:
        continue
    events = [{"sample": int(ev["sample"]), "label": int(ev["label"])} if isinstance(ev, dict) else {"sample": int(ev[0]), "label": int(ev[1])} for ev in events]

    n_channels, n_total = eeg.shape
    n_samples_epoch = int((epoch_tend - epoch_tstart) * sampling_rate)
    epoch_start_offset = int(epoch_tstart * sampling_rate)
    for ev in events:
        s0 = ev["sample"] + epoch_start_offset
        s1 = s0 + n_samples_epoch
        if s0 < 0 or s1 > n_total:
            continue
        epochs_list.append(eeg[:, s0:s1].copy())
        labels_list.append(ev["label"])

if not epochs_list:
    raise FileNotFoundError(f"No data found in {folder_path}. Run run_vep.py with calibration_mode=True first.")

combined_epochs = np.stack(epochs_list, axis=0)
combined_labels = np.array(labels_list, dtype=int)

print("Combined shape:", combined_epochs.shape, "labels:", combined_labels.shape)

# baseline duration, then crop to exclude baseline period (no subtract)
# baseline_average = np.mean(combined_epochs[:, :, :baseline_samples], axis=2, keepdims=True)
# baseline_corrected = combined_epochs - baseline_average
# cropped_epochs = combined_epochs[:, :, baseline_samples:]
cropped_epochs = combined_epochs[:, :, baseline_samples:]

def run_motor_csp_lda(epochs, labels, test_size=0.25, random_state=42, print_acc=True):
    X_train, X_test, y_train, y_test = train_test_split(
        epochs, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    # Same as original run_fbtrca: mean subtract over time
    X_train = X_train - np.mean(X_train, axis=-1, keepdims=True)
    X_test = X_test - np.mean(X_test, axis=-1, keepdims=True)
    n_components = min(4, X_train.shape[1], X_train.shape[2] - 1)
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat = csp.transform(X_test)
    model = LinearDiscriminantAnalysis()
    model.fit(X_train_feat, y_train)
    pred_labels = model.predict(X_test_feat)
    acc = accuracy_score(y_test, pred_labels)
    if print_acc:
        print("Model: CSP+LDA  Acc: {:.2f}".format(acc))
    return confusion_matrix(y_test, pred_labels, normalize='true'), acc, model, csp

cm, acc, model, csp = run_motor_csp_lda(cropped_epochs, combined_labels, print_acc=True)

os.makedirs(model_save_dir, exist_ok=True)
with open(model_save_dir + model_name, 'wb') as f:
    pickle.dump(model, f)
with open(model_save_dir + csp_name, 'wb') as f:
    pickle.dump(csp, f)
