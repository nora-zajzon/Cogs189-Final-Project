"""
EEG BCI Maze Navigation – data acquisition and real-time inference.
Motor imagery: Class 0 = left hand, Class 1 = right foot.
- Training mode (calibration_mode=True): display cue, record epochs with event markers.
- Real-time mode: sliding-window classification, prediction smoothing; no maze yet.
"""
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
sampling_rate = 250
cyton_in = True
width = 1536
height = 864
subject = 1
session = 1
calibration_mode = True  # True = collect labeled data; False = real-time prediction
n_per_class = 30  # trials per class (left hand, right foot)
run = 1
# Motor imagery trial timing (s)
baseline_duration = 1.5
record_duration = 3.0
# Real-time
realtime_window_sec = 1.5
realtime_update_interval_sec = 0.25
smoothing_n_predictions = 5
# Paths
save_dir = f"data/motor_imagery_2class/sub-{subject:02d}/ses-{session:02d}/"
model_save_dir = "cache"
model_name = "motor_lda_model.pkl"
csp_name = "motor_csp.pkl"

os.makedirs(save_dir, exist_ok=True)
save_file_eeg = os.path.join(save_dir, f"eeg_run-{run}.npy")
save_file_events = os.path.join(save_dir, f"events_run-{run}.npy")
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
board = None
stop_event = None
queue_in = None


def find_openbci_port():
    import glob
    from serial import Serial
    BAUD_RATE = 115200
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        ports = glob.glob("/dev/ttyUSB*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/cu.usbserial*")
    else:
        raise EnvironmentError("Unsupported OS for port detection")
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b"v")
            time.sleep(2)
            line = ""
            if s.in_waiting:
                while "$$$" not in line:
                    c = s.read().decode("utf-8", errors="replace")
                    line += c
                if "OpenBCI" in line:
                    s.close()
                    return port
            s.close()
        except Exception:
            pass
    raise OSError("Cannot find OpenBCI port.")


def start_board():
    global board, stop_event, queue_in
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from threading import Thread, Event
    from queue import Queue

    CYTON_BOARD_ID = 0
    ANALOGUE_MODE = "/2"
    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board("/0")
    board.config_board("//")
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)
    stop_event = Event()
    queue_in = Queue()

    def get_data(q):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            ts = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(ts) > 0:
                q.put((eeg_in, aux_in, ts))
            time.sleep(0.1)

    t = Thread(target=get_data, args=(queue_in,))
    t.daemon = True
    t.start()
    return board


def stop_board():
    global board, stop_event
    if stop_event is not None:
        stop_event.set()
    if board is not None:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass


def drain_queue_into(eeg, aux, timestamp):
    """Append all available chunks from queue; return (eeg, aux, timestamp)."""
    while not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg = np.hstack((eeg, eeg_in)) if eeg.size else eeg_in
        aux = np.hstack((aux, aux_in)) if aux.size else aux_in
        timestamp = np.concatenate((timestamp, ts_in)) if timestamp.size else ts_in
    return eeg, aux, timestamp


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
    events = []  # list of {"sample": int, "label": 0|1}

    instruction_stim = visual.TextStim(
        window, text="", pos=(0, 0), color="white", units="norm", height=0.08, alignText="center"
    )
    trial_stim = visual.TextStim(
        window, text="", pos=(0, -0.85), color="gray", units="norm", height=0.05
    )

    for i_trial, label in enumerate(trial_sequence):
        # Flush queue so we know current buffer size when we show cue
        eeg, aux, timestamp = drain_queue_into(eeg, aux, timestamp)

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

        # Drain again so next trial has up-to-date buffer
        eeg, aux, timestamp = drain_queue_into(eeg, aux, timestamp)

        keys = keyboard.getKeys()
        if "escape" in keys:
            break

    # Final drain
    eeg, aux, timestamp = drain_queue_into(eeg, aux, timestamp)
    np.save(save_file_eeg, eeg)
    # Save events as list of dicts (numpy will store as object array)
    np.save(save_file_events, np.array(events, dtype=object))
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


def load_realtime_pipeline():
    """Load CSP and LDA from disk; return (csp, model) or (None, None)."""
    if not os.path.exists(model_file_path) or not os.path.exists(csp_file_path):
        return None, None
    with open(csp_file_path, "rb") as f:
        csp = pickle.load(f)
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
    return csp, model


def run_realtime():
    csp, model = load_realtime_pipeline()
    if model is None:
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
        eeg_buffer, _, _ = drain_queue_into(eeg_buffer, np.zeros((3, 0)), np.zeros((0)))
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
            break


# --- Main ---
if __name__ == "__main__":
    if cyton_in:
        start_board()
    try:
        if calibration_mode:
            run_calibration()
        else:
            run_realtime()
    finally:
        if cyton_in:
            stop_board()
    window.close()
