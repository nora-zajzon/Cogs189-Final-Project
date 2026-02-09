from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
import random
import os
import pickle
import sys
import time
import mne

# --- Params ---
cyton_in = True
lsl_out = False
width = 1536
height = 864
subject = 1
session = 1
calibration_mode = False  # True = collect labeled data; False = real-time prediction
n_per_class = 30  # trials per class (left hand, right foot)
run = 1  # Run number, it is used as the random seed for the trial sequence generation
# Motor imagery trial timing (s)
baseline_duration = 1.5
record_duration = 3.0
# Real-time
realtime_window_sec = 1.5
realtime_update_interval_sec = 0.25
smoothing_n_predictions = 5
# Paths
save_dir = f'data/motor_imagery_2class/sub-{subject:02d}/ses-{session:02d}/'  # Directory to save data to
save_file_eeg = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_timestamp = save_dir + f'timestamp_{n_per_class}-per-class_run-{run}.npy'
save_file_metadata = save_dir + f'metadata_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_events = save_dir + f'events_{n_per_class}-per-class_run-{run}.npy'
model_save_dir = "cache"
model_name = "motor_lda_model.pkl"
csp_name = "motor_csp.pkl"
model_file_path = os.path.join(model_save_dir, model_name)
csp_file_path = os.path.join(model_save_dir, csp_name)

# Class labels and labels for display
CLASS_LEFT_HAND = 0
CLASS_RIGHT_FOOT = 1
INSTRUCTION_TEXT = {CLASS_LEFT_HAND: "Move LEFT HAND", CLASS_RIGHT_FOOT: "Move RIGHT FOOT"}

keyboard = keyboard.Keyboard()
window = visual.Window(
    size=[width, height],
    checkTiming=True,
    allowGUI=False,
    fullscr=False,
    useRetina=False,
)

# --- EEG acquisition (BrainFlow / Cyton) ---
if cyton_in:
    import glob, sys, time, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue
    sampling_rate = 250
    CYTON_BOARD_ID = 0  # 0 if no daisy 2 if use daisy board, 6 if using daisy+wifi shield
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2'  # Reads from analog pins A5(D11), A6(D12) and if no 
                          # wifi shield is present, then A7(D13) as well.
    def find_openbci_port():
        """Finds the port to which the Cyton Dongle is connected to."""
        # Find serial port names per OS
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        openbci_port = ''
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                line = ''
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    c = ''
                    while '$$$' not in line:
                        c = s.read().decode('utf-8', errors='replace')
                        line += c
                    if 'OpenBCI' in line:
                        openbci_port = port
                s.close()
            except (OSError, serial.SerialException):
                pass
        if openbci_port == '':
            raise OSError('Cannot find OpenBCI port.')
            exit()
        else:
            return openbci_port
        
    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    res_query = board.config_board('/0')
    print(res_query)
    res_query = board.config_board('//')
    print(res_query)
    res_query = board.config_board(ANALOGUE_MODE)
    print(res_query)
    board.start_stream(45000)
    stop_event = Event()
    
    def get_data(queue_in, lsl_out=False):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)
    
    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None
    if os.path.exists(csp_file_path):
        with open(csp_file_path, 'rb') as f:
            csp = pickle.load(f)
    else:
        csp = None
else:
    board = None
    stop_event = None
    queue_in = None
    model = None
    csp = None


# --- Trial sequence: 2 classes, n_per_class each ---
def build_trial_sequence(n_per_class, seed=0):
    seq = [CLASS_LEFT_HAND] * n_per_class + [CLASS_RIGHT_FOOT] * n_per_class
    random.seed(seed)
    random.shuffle(seq)
    return seq


# --- Training mode: collect labeled data with event markers ---
def run_calibration():
    trial_sequence = build_trial_sequence(n_per_class, seed=run)
    eeg = np.zeros((8, 0))
    aux = np.zeros((3, 0))
    timestamp = np.zeros((0))
    eeg_trials = []
    aux_trials = []
    events = []  # list of {"sample": int, "label": 0|1}

    instruction_stim = visual.TextStim(
        window, text="", pos=(0, 0), color="white", units="norm", height=0.08, alignText="center"
    )
    trial_stim = visual.TextStim(
        window, text="", pos=(0, -0.85), color="gray", units="norm", height=0.05
    )

    for i_trial, label in enumerate(trial_sequence):
        # Collect all data from the queue
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

        instruction_stim.text = INSTRUCTION_TEXT[label]
        trial_stim.text = f"Trial {i_trial + 1} / {len(trial_sequence)}"
        instruction_stim.draw()
        trial_stim.draw()
        window.flip()
        # Event: cue onset at current sample index
        cue_sample = eeg.shape[1]
        events.append({"sample": int(cue_sample), "label": int(label)})

        core.wait(baseline_duration)
        core.wait(record_duration)

        # Collect data again after trial
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

        # Extract trial data
        print('total: ', eeg.shape, aux.shape, timestamp.shape)
        baseline_duration_samples = int(baseline_duration * sampling_rate)
        trial_duration_samples = int(record_duration * sampling_rate)
        trial_start = max(0, cue_sample - baseline_duration_samples)
        trial_end_required = trial_start + baseline_duration_samples + trial_duration_samples
        if trial_end_required > eeg.shape[1]:
            print(f'Warning: Not enough data for trial {i_trial}, skipping trial extraction')
            continue
        filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=8, h_freq=30, verbose=False)
        trial_eeg = np.copy(filtered_eeg[:, trial_start:trial_end_required])
        trial_aux = np.copy(aux[:, trial_start:trial_end_required])
        print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
        baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True)
        trial_eeg -= baseline_average
        eeg_trials.append(trial_eeg)
        aux_trials.append(trial_aux)

        keys = keyboard.getKeys()
        if "escape" in keys:
            if cyton_in:
                os.makedirs(save_dir, exist_ok=True)
                np.save(save_file_eeg, eeg)
                np.save(save_file_aux, aux)
                np.save(save_file_eeg_trials, eeg_trials)
                np.save(save_file_aux_trials, aux_trials)
                np.save(save_file_events, np.array(events, dtype=object))
                stop_event.set()
                board.stop_stream()
                board.release_session()
            core.quit()

    # Final collection
    while not queue_in.empty():
        eeg_in, aux_in, timestamp_in = queue_in.get()
        print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
        eeg = np.concatenate((eeg, eeg_in), axis=1)
        aux = np.concatenate((aux, aux_in), axis=1)
        timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

    if cyton_in:
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_file_eeg, eeg)
        np.save(save_file_aux, aux)
        np.save(save_file_eeg_trials, eeg_trials)
        np.save(save_file_aux_trials, aux_trials)
        np.save(save_file_events, np.array(events, dtype=object))
        board.stop_stream()
        board.release_session()
    print(f"Saved eeg shape {eeg.shape}, {len(events)} events to {save_dir}")


# --- Real-time mode: load model + CSP, sliding window, smoothing ---
def _preprocess_chunk(chunk, sfreq):
    """Bandpass 8-30 Hz, notch 60 Hz, baseline subtract. chunk: (n_ch, n_samples)."""
    x = mne.filter.notch_filter(chunk, Fs=sfreq, freqs=60.0, verbose=False)
    x = mne.filter.filter_data(x, sfreq=sfreq, l_freq=8, h_freq=30, verbose=False)
    n_baseline = min(int(0.5 * sfreq), x.shape[1] // 5)
    if n_baseline > 0:
        x = x - np.mean(x[:, :n_baseline], axis=1, keepdims=True)
    return x


def run_realtime():
    if model is None or csp is None:
        print("No model found. Run training first (scripts/train_motor.py).")
        return

    n_ch = 8
    window_samples = int(realtime_window_sec * sampling_rate)
    eeg_buffer = np.zeros((n_ch, 0))
    prediction_history = []
    last_update_time = 0

    instruction_stim = visual.TextStim(
        window, text="Real-time: think LEFT HAND or RIGHT FOOT", pos=(0, 0.2),
        color="white", units="norm", height=0.06
    )
    pred_stim = visual.TextStim(
        window, text="Pred: --", pos=(0, -0.2), color="lime", units="norm", height=0.1
    )

    while True:
        # Collect data from queue
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            eeg_buffer = np.concatenate((eeg_buffer, eeg_in), axis=1)
            if eeg_buffer.shape[1] > 2 * window_samples:
                eeg_buffer = eeg_buffer[:, -2 * window_samples:]

        now = core.getTime()
        if now - last_update_time >= realtime_update_interval_sec and eeg_buffer.shape[1] >= window_samples:
            last_update_time = now
            chunk = eeg_buffer[:, -window_samples:]
            ep = _preprocess_chunk(chunk, sampling_rate)
            ep = ep[np.newaxis, ...]
            try:
                feat = csp.transform(ep)
                pred = model.predict(feat)[0]
                prediction_history.append(pred)
                if len(prediction_history) > smoothing_n_predictions:
                    prediction_history.pop(0)
                smooth_pred = int(np.round(np.mean(prediction_history)))
                pred_stim.text = f"Pred: {'LEFT' if smooth_pred == 0 else 'RIGHT'}  (raw: {pred})"
            except Exception as e:
                pred_stim.text = f"Pred: err ({e})"

        instruction_stim.draw()
        pred_stim.draw()
        window.flip()

        keys = keyboard.getKeys()
        if "escape" in keys:
            stop_event.set()
            board.stop_stream()
            board.release_session()
            core.quit()


# --- Main ---
if __name__ == "__main__":
    if calibration_mode:
        run_calibration()
    else:
        run_realtime()
    window.close()
