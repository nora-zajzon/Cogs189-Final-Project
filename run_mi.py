from psychopy import visual, core, event
from psychopy.hardware import keyboard
import numpy as np
import random
import os
import pickle
import sys
import time
import mne

# --- Params ---
cyton_in = True # eeg input (set to false for fake data)
sampling_rate = 250  # for debugging
lsl_out = False
width = 1536
height = 864
subject = 1
session = 4
data_gathered = True  # False = collect labeled data; True = play snake game with BCI
n_per_class = 20  # trials per class (left hand, right foot)
run = 1  # Run number, it is used as the random seed for the trial sequence generation
# Motor imagery trial timing (s)
baseline_duration = 0.5 # seconds of baseline before recording
record_duration = 1.5 # seconds of motor imagery recording per trial
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
model_save_dir = "./scripts/cache"
model_name = "motor_lda_model.pkl"
csp_name = "motor_csp.pkl"
model_file_path = os.path.join(model_save_dir, model_name)
csp_file_path = os.path.join(model_save_dir, csp_name)

# Class labels and labels for display
CLASS_LEFT_HAND = 0
CLASS_RIGHT_FOOT = 1
INSTRUCTION_TEXT = {CLASS_LEFT_HAND: "Move LEFT HAND", CLASS_RIGHT_FOOT: "Move RIGHT HAND"}

kb = keyboard.Keyboard()
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
# else: # fake data for debugging (uncomment to use for debugging and comment out the above)
#     board = None
#     stop_event = None
#     model = None
#     csp = None

    # from queue import Queue
    # queue_in = Queue()

    # # simulate fake EEG data in background
    # import threading
    # def fake_eeg_stream():
    #     sampling_rate = 250
    #     toggle = 0

    #     while True:
    #         eeg = np.random.randn(8, sampling_rate // 10) * 3

    #         # Inject class-like structure
    #         if toggle == 0:
    #             eeg[2] += 2   # channel 3 boosted
    #         else:
    #             eeg[5] += 2   # channel 6 boosted

    #         toggle = 1 - toggle

    #         aux = np.random.randn(3, sampling_rate // 10)
    #         timestamp = np.arange(eeg.shape[1])
    #         queue_in.put((eeg, aux, timestamp))
    #         time.sleep(0.1)

    # threading.Thread(target=fake_eeg_stream, daemon=True).start()



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
    countdown_stim = visual.TextStim(
        window, text="", pos=(0, 0.25), color="yellow", units="norm", height=0.15, alignText="center"
    )
    recording_stim = visual.TextStim(
        window, text="RECORDING", pos=(0, 0.25), color="red", units="norm", height=0.1, alignText="center"
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

        trial_stim.text = f"Trial {i_trial + 1} / {len(trial_sequence)}"

        # Preparation prompt
        instruction_stim.text = f"Prepare: {INSTRUCTION_TEXT[label]}"
        instruction_stim.draw()
        trial_stim.draw()
        window.flip()
        core.wait(1.5)

        # Countdown
        for count in [3, 2, 1]:
            countdown_stim.text = str(count)
            instruction_stim.draw()
            countdown_stim.draw()
            trial_stim.draw()
            window.flip()
            core.wait(0.5)

        # Collect queued data right before recording onset
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get()
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

        # Cue onset sample index (event appended only after successful extraction)
        cue_sample = eeg.shape[1]

        # Recording phase
        instruction_stim.text = INSTRUCTION_TEXT[label]
        recording_stim.draw()
        instruction_stim.draw()
        trial_stim.draw()
        window.flip()
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
        events.append({"sample": int(cue_sample), "label": int(label)})

        keys = kb.getKeys()
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

    if True:
        # Always save data
        os.makedirs(save_dir, exist_ok=True)

        np.save(save_file_eeg, eeg)
        np.save(save_file_aux, aux)
        np.save(save_file_eeg_trials, eeg_trials)
        np.save(save_file_aux_trials, aux_trials)
        np.save(save_file_events, np.array(events, dtype=object))

        # Only shut down hardware if using Cyton
        if cyton_in:
            stop_event.set()
            board.stop_stream()
            board.release_session()
            print(f"Saved eeg shape {eeg.shape}, {len(events)} events to {save_dir}")


# --- Real-time mode: load model + CSP, sliding window, smoothing ---
# def _preprocess_chunk(chunk, sfreq):
#     """Bandpass 8-30 Hz, notch 60 Hz, baseline subtract. chunk: (n_ch, n_samples)."""
#     x = mne.filter.notch_filter(chunk, Fs=sfreq, freqs=60.0, verbose=False)
#     x = mne.filter.filter_data(x, sfreq=sfreq, l_freq=8, h_freq=30, verbose=False)
#     n_baseline = min(int(0.5 * sfreq), x.shape[1] // 5)
#     if n_baseline > 0:
#         x = x - np.mean(x[:, :n_baseline], axis=1, keepdims=True)
#     return x

def _preprocess_chunk(chunk, sfreq):

    x = mne.filter.filter_data(
        chunk,
        sfreq=sfreq,
        l_freq=8,
        h_freq=30,
        verbose=False
    )

    x = x - np.mean(x, axis=1, keepdims=True)

    return x

class SnakeGame:
    """
    Snake game with discrete LEFT/RIGHT decision control.

    Behavior:
    - Snake moves straight automatically.
    - When snake head aligns with apple on x OR y axis,
      the snake stops and waits for a turn decision.
    - External loop must call:
          game.set_direction(...)
          game.resolve_turn()
      after countdown + 1.5s decision window.
    """

    LEFT = 'left'
    RIGHT = 'right'

    _UP = (0, 1)
    _DOWN = (0, -1)
    _LEFT = (-1, 0)
    _RIGHT = (1, 0)

    def __init__(self, psychopy_window, grid_n=20, cell_px=30, move_interval=0.35):
        self.win = psychopy_window
        self.grid_n = grid_n
        self.cell_px = cell_px
        self.move_interval = move_interval

        # Turn lookup tables
        self._TURN_LEFT = {
            self._UP: self._LEFT,
            self._LEFT: self._DOWN,
            self._DOWN: self._RIGHT,
            self._RIGHT: self._UP,
        }
        self._TURN_RIGHT = {
            self._UP: self._RIGHT,
            self._RIGHT: self._DOWN,
            self._DOWN: self._LEFT,
            self._LEFT: self._UP,
        }

        # Visual objects (created once)
        self._cell = visual.Rect(
            self.win,
            width=cell_px - 2,
            height=cell_px - 2,
            units='pix'
        )

        self._border = visual.Rect(
            self.win,
            width=grid_n * cell_px + 4,
            height=grid_n * cell_px + 4,
            units='pix',
            fillColor="#183B49",
            lineColor='#183B49',
        )

        self.score_txt = visual.TextStim(
            self.win,
            text='Score: 0',
            pos=(0, grid_n * cell_px / 2 + 30),
            color='white',
            units='pix',
            height=24,
        )

        self.reset()

    # ==========================================================
    # Reset
    # ==========================================================
    def reset(self):
        mid = self.grid_n // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self._snake_set = set(self.snake)

        self._heading = self._RIGHT
        self.food = self._place_food()

        self.score = 0
        self.alive = True

        self._pending = None
        self._waiting_for_turn = False
        self._last_move_t = core.getTime()

    # ==========================================================
    # External Control API
    # ==========================================================
    def set_direction(self, direction):
        """Queue a LEFT or RIGHT turn decision."""
        if direction in (self.LEFT, self.RIGHT):
            self._pending = direction

    def resolve_turn(self):
        """
        Apply pending turn and resume movement.
        Call this AFTER countdown + decision window.
        """
        if self._pending == self.LEFT:
            self._heading = self._TURN_LEFT[self._heading]
        elif self._pending == self.RIGHT:
            self._heading = self._TURN_RIGHT[self._heading]

        self._pending = None
        self._waiting_for_turn = False

    # ==========================================================
    # Game Update
    # ==========================================================
    def tick(self):
        """
        Advance game state.
        - Moves automatically.
        - Stops if alignment condition is met.
        """
        if not self.alive:
            return False

        now = core.getTime()
        if now - self._last_move_t < self.move_interval:
            return True
        self._last_move_t = now

        # If waiting for decision, do not move
        if self._waiting_for_turn:
            return True

        head_x, head_y = self.snake[0]
        apple_x, apple_y = self.food

        # Stop if aligned with apple (same row or column)
        # Determine if snake is aligned
        aligned = (head_x == apple_x) or (head_y == apple_y)

        if aligned:

            dx, dy = self._heading

            # Vector from head to apple
            vec_x = apple_x - head_x
            vec_y = apple_y - head_y

            # Check if currently heading toward apple
            heading_correct = (
                (dx != 0 and np.sign(vec_x) == dx) or
                (dy != 0 and np.sign(vec_y) == dy)
            )

            if not heading_correct:
                self._waiting_for_turn = True
                return True

        # Normal forward movement
        new_head = (
            (head_x + self._heading[0]) % self.grid_n,
            (head_y + self._heading[1]) % self.grid_n,
        )

        # Collision with self
        if new_head in self._snake_set:
            self.alive = False
            return False

        self.snake.insert(0, new_head)
        self._snake_set.add(new_head)

        # Eat apple
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.score_txt.text = f'Score: {self.score}'
        else:
            removed = self.snake.pop()
            self._snake_set.discard(removed)

        return True

    # ==========================================================
    # Draw
    # ==========================================================
    def draw(self):
        """Render the game."""
        ox = -self.grid_n * self.cell_px / 2
        oy = -self.grid_n * self.cell_px / 2

        self._border.draw()

        # Draw food
        self._cell.pos = (
            ox + self.food[0] * self.cell_px + self.cell_px / 2,
            oy + self.food[1] * self.cell_px + self.cell_px / 2
        )
        self._cell.fillColor = '#F5F0E6'
        self._cell.lineColor = '#F5F0E6'
        self._cell.draw()

        # Draw snake (tail first)
        for i in range(len(self.snake) - 1, -1, -1):
            seg = self.snake[i]
            self._cell.pos = (
                ox + seg[0] * self.cell_px + self.cell_px / 2,
                oy + seg[1] * self.cell_px + self.cell_px / 2
            )
            self._cell.fillColor = '#C69214' if i == 0 else '#FFCD00'
            self._cell.lineColor = self._cell.fillColor
            self._cell.draw()

        self.score_txt.draw()

    # ==========================================================
    # Food Placement
    # ==========================================================
    def _place_food(self):
        while True:
            pos = (
                random.randint(0, self.grid_n - 1),
                random.randint(0, self.grid_n - 1)
            )
            if pos not in self._snake_set:
                return pos

def run_snake_game():
    global model, csp

    # Load model if needed
    if model is None and os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
    if csp is None and os.path.exists(csp_file_path):
        with open(csp_file_path, 'rb') as f:
            csp = pickle.load(f)

    if model is None or csp is None:
        print("No model found. Run training first (scripts/train_motor.py).")
        return

    game = SnakeGame(window)

    n_ch = 8
    win_samp = int(realtime_window_sec * sampling_rate)
    eeg_buf = np.zeros((n_ch, 0))

    while game.alive:

        # =========================
        # Drain EEG queue
        # =========================
        while not queue_in.empty():
            eeg_in, _, _ = queue_in.get()
            eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)

            # keep buffer bounded
            if eeg_buf.shape[1] > 2 * win_samp:
                eeg_buf = eeg_buf[:, -2 * win_samp:]

        # =========================
        # DECISION PHASE
        # =========================
        if getattr(game, "_waiting_for_turn", False):

            # Countdown
            for count in [3, 2, 1]:
                cd = visual.TextStim(
                    window,
                    text=str(count),
                    color='yellow',
                    height=80,
                    units='pix'
                )
                game.draw()
                cd.draw()
                window.flip()
                core.wait(0.5)

            # 1.5 second decision window
            decision_start = core.getTime()
            pred_hist = []

            while core.getTime() - decision_start < 1.5:

                while not queue_in.empty():
                    eeg_in, _, _ = queue_in.get()
                    eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)

                if eeg_buf.shape[1] >= win_samp:
                    chunk = eeg_buf[:, -win_samp:]
                    ep = _preprocess_chunk(chunk, sampling_rate)[np.newaxis, ...]

                    try:
                        feat = csp.transform(ep)
                        pred = model.predict(feat)[0]
                        pred_hist.append(pred)
                    except:
                        pass

                game.draw()
                window.flip()

            # Smooth prediction
            if len(pred_hist) > 0:
                decision = int(np.round(np.mean(pred_hist)))

                if decision == CLASS_LEFT_HAND:
                    game.set_direction(SnakeGame.LEFT)
                else:
                    game.set_direction(SnakeGame.RIGHT)

            # Apply turn + resume movement
            game.resolve_turn()

        # =========================
        # Normal movement
        # =========================
        game.tick()
        game.draw()
        window.flip()

        if 'escape' in kb.getKeys():
            break

    # Game over screen
    over_txt = visual.TextStim(
        window,
        text=f'GAME OVER\nScore: {game.score}\n\nPress ESC to exit',
        pos=(0, 0),
        color='white',
        units='pix',
        height=30,
        wrapWidth=500,
    )
    over_txt.draw()
    window.flip()

    while True:
        if 'escape' in kb.getKeys():
            break
        core.wait(0.05)

    if cyton_in:
        stop_event.set()
        board.stop_stream()
        board.release_session()



def run_snake_game_keyboard():

    game = SnakeGame(window)

    info_txt = visual.TextStim(
        window,
        text="Keyboard Test Mode\n← = LEFT   → = RIGHT\nESC = quit",
        pos=(0, -(game.grid_n * game.cell_px / 2 + 60)),
        color='white',
        units='pix',
        height=20,
    )

    while game.alive:

        # =========================
        # DECISION PHASE
        # =========================
        if getattr(game, "_waiting_for_turn", False):

            # Countdown
            for count in [3, 2, 1]:
                cd = visual.TextStim(
                    window,
                    text=str(count),
                    color='yellow',
                    height=80,
                    units='pix'
                )
                game.draw()
                cd.draw()
                info_txt.draw()
                window.flip()
                core.wait(0.5)

            # 1.5 second decision window
            decision_start = core.getTime()
            chosen = None

            while core.getTime() - decision_start < 1.5:
                keys = event.getKeys()

                if 'left' in keys:
                    chosen = SnakeGame.LEFT
                if 'right' in keys:
                    chosen = SnakeGame.RIGHT

                if 'escape' in keys:
                    return

                game.draw()
                info_txt.draw()
                window.flip()

            if chosen is not None:
                game.set_direction(chosen)

            game.resolve_turn()

        # =========================
        # Normal movement
        # =========================
        keys = event.getKeys()
        if 'escape' in keys:
            return

        game.tick()
        game.draw()
        info_txt.draw()
        window.flip()

        core.wait(0.01)

if __name__ == "__main__":
    if not data_gathered:
        run_calibration()
    else:
        run_snake_game()

        # run_snake_game_keyboard()
