import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

# -------- CONFIG --------
base_path = '../data/motor_imagery_2class/sub-01/'
sessions = ['ses-01']
model_save_dir = '../cache/'
model_name = 'motor_lda_model.pkl'
csp_name = 'motor_csp.pkl'

sampling_rate = 250
baseline_duration = 0.2
baseline_samples = int(baseline_duration * sampling_rate)

# -------- LOAD DATA --------
all_epochs = []
all_labels = []

for session in sessions:

    folder_path = os.path.join(base_path, session)

    if not os.path.exists(folder_path):
        continue

    run_files = [
        f for f in os.listdir(folder_path)
        if f.startswith('eeg-trials_') and f.endswith('.npy')
    ]

    print(f"\nSession {session} — {len(run_files)} runs")

    for run_file in run_files:

        eeg_trials = np.load(os.path.join(folder_path, run_file))
        events = np.load(
            os.path.join(folder_path,
                         f'events_{run_file.split("eeg-trials_")[1]}'),
            allow_pickle=True
        )

        labels = np.array([ev["label"] for ev in events], dtype=int)

        print("Loaded:", run_file, eeg_trials.shape)

        all_epochs.append(eeg_trials)
        all_labels.append(labels)

epochs = np.concatenate(all_epochs, axis=0)
labels = np.concatenate(all_labels, axis=0)

print("\nFINAL SHAPE:", epochs.shape)
print("CLASS BALANCE:", np.bincount(labels))

# Remove baseline
epochs = epochs[:, :, baseline_samples:]
print("Per-channel variance:", np.var(epochs, axis=(0,2)))
# -------- TRAIN --------

def run_motor_csp_lda(epochs, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        epochs,
        labels,
        test_size=0.25,
        stratify=labels,
        random_state=42
    )


    # 🔥 3-class CSP
    n_components = 6   # 2 per class

    csp = CSP(
        n_components=n_components,
        reg='oas',      # better shrinkage
        log=True,
        norm_trace=False
    )

    X_train_feat = csp.fit_transform(X_train, y_train)
    X_test_feat = csp.transform(X_test)

    # LDA works perfectly fine for 3-class
    model = LinearDiscriminantAnalysis()
    model.fit(X_train_feat, y_train)

    preds = model.predict(X_test_feat)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, normalize='true')

    print(f"\nAccuracy: {acc:.3f}")
    print("Confusion matrix:\n", cm)

    return model, csp

model, csp = run_motor_csp_lda(epochs, labels)

# -------- SAVE --------
os.makedirs(model_save_dir, exist_ok=True)

with open(os.path.join(model_save_dir, model_name), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(model_save_dir, csp_name), 'wb') as f:
    pickle.dump(csp, f)

print("\nSaved model + CSP")
