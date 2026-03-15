from psychopy import visual, core, event # PsychoPy: (visual: drawing text, rectangles, windows; core: timing utilities; event: keyboard input)
from psychopy.hardware import keyboard # keyboard api
import numpy as np
import random
import os
import pickle # save/load trained model and CSP object
import sys
import time
import mne # EEG filtering

# --- Params ---
cyton_in = False # OpenBCI cyton eeg input (set to false for fake data)
sampling_rate = 250  # how often EEG device measures brain signal per second
lsl_out = False # config for Lab Streaming Layer
width = 1536 # window width
height = 864 # window height
subject = 1
session = 4
data_gathered = True  # False = collect labeled data; True = play snake game with BCI
n_per_class = 20  # trials per class (left hand, right foot)
run = 1  # Run number, it is used as the random seed for the trial sequence generation
# Motor imagery trial timing (s)
baseline_duration = 0.5 # seconds of baseline before imagery period
record_duration = 1.5 # seconds of motor imagery recording per trial
# Real-time
realtime_window_sec = 1.5 # in real-time, classifier looks at most recent 1.5 seconds of EEG

# Paths
save_dir = f'data/motor_imagery_2class/sub-{subject:02d}/ses-{session:02d}/'  # Directory to save data to
save_file_eeg = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_events = save_dir + f'events_{n_per_class}-per-class_run-{run}.npy'
model_save_dir = "./scripts/cache" # where trained model and CSP are loaded from
model_name = "motor_lda_model.pkl"
csp_name = "motor_csp.pkl"
model_file_path = os.path.join(model_save_dir, model_name)
csp_file_path = os.path.join(model_save_dir, csp_name)

# Class labels and labels for display
CLASS_LEFT_HAND = 0
CLASS_RIGHT_HAND = 1
INSTRUCTION_TEXT = {CLASS_LEFT_HAND: "Move LEFT HAND", CLASS_RIGHT_HAND: "Move RIGHT HAND"}

# PsychoPy keyboard input object
kb = keyboard.Keyboard()
# visual window initialization
window = visual.Window(
    size=[width, height],
    allowGUI=False,
    fullscr=False,
)

# --- EEG acquisition (BrainFlow / Cyton) ---
if cyton_in:
    import glob, sys, time, serial # to find and communicate with the Cyton USB dongle
    from brainflow.board_shim import BoardShim, BrainFlowInputParams # SDK to talk to EEG boards
    from serial import Serial # to detect Cyton port
    from threading import Thread, Event # background threading for EEG acquisition
    from queue import Queue # thread-safe (prevent race conditions) data buffer
    CYTON_BOARD_ID = 0  # 0 if no daisy 2 if use daisy board, 6 if using daisy+wifi shield
    BAUD_RATE = 115200 # data transmission speed
    ANALOGUE_MODE = '/2'  # Reads from analog pins A5(D11), A6(D12) and if no 
                          # wifi shield is present, then A7(D13) as well.
    
    def find_openbci_port():
        """Finds the port to which the Cyton Dongle is connected to."""
        # Find serial port names per OS
        if sys.platform.startswith('win'): # windows
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'): # mac
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
        
    print(BoardShim.get_board_descr(CYTON_BOARD_ID)) # Prints cyton info abuot EEG channels
    params = BrainFlowInputParams() # settings for connecting the board
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
        """
        One thread pulls EEG data from the Cyton Board and pushes it into queue,
        while main thread runs the game and consumes that data
        """

        while not stop_event.is_set():
            data_in = board.get_board_data() # shape is [channels, samples]
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)] # extract timestamps
            eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)] # extract EEG channels (8 EEG channels)
            aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)] # auxiliary non-eeg channels
            if len(timestamp_in) > 0:
                print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                queue_in.put((eeg_in, aux_in, timestamp_in)) # queue used prevent race conditions
            time.sleep(0.1) # each loop grabs ~25 samples
    
    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out)) # thread for continuously reading EEG data and pushing it into queue
    cyton_thread.daemon = True # if program exists, thread will automatically stop
    cyton_thread.start()

    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f) # LDA classifier if it exists 
    # Linear Discriminant Analysis (LDA): finds linear boundary separating two classes
    # - LDA tries maximizing distance between class centers and keeps each class cluster tight
    
    else:
        model = None
    if os.path.exists(csp_file_path):
        with open(csp_file_path, 'rb') as f:
            csp = pickle.load(f) # trained CSP object if it exists 
    
    # Common Spatial Patterns (CSP): transforms EEG signals into discriminative features before classification
    # - Learns linear combinations of electrodes
    # - tries to maximize variance for class A and minimize variance for class B
    # - filters amplify the difference between classes
    # - CSP learns combinations of electrodes that make one class show strong variance and the other class show weak variance 
    # - it turns raw multichannel EEG into a smaller set of much more useful features

    else:
        csp = None
else: # if not using live EEG
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
    #         eeg = np.random.randn(8, sampling_rate // 10) * 3 # generates random numbers from a normal distribution similar to EEG

    #         # Inject class-like structure
    #         if toggle == 0:
    #             eeg[2] += 2   # channel 3 boosted for LEFT imagery
    #         else:
    #             eeg[5] += 2   # channel 6 boosted for RIGHT imagery

    #         toggle = 1 - toggle

    #         aux = np.random.randn(3, sampling_rate // 10) # fake auxiliary data
    #         timestamp = np.arange(eeg.shape[1])
    #         queue_in.put((eeg, aux, timestamp))
    #         time.sleep(0.1)

    # threading.Thread(target=fake_eeg_stream, daemon=True).start()



# Trial sequence: 2 classes, n_per_class each, randomly shuffled
def build_trial_sequence(n_per_class, seed=0): 
    seq = [CLASS_LEFT_HAND] * n_per_class + [CLASS_RIGHT_HAND] * n_per_class
    random.seed(seed)
    random.shuffle(seq)
    return seq



# Training mode: collect labeled data with event markers
def run_calibration():
    """
    Presents a sequence of motor-imagery prompts, collects EEG data during each trial,
    extract the corresponding EEG segment, labels it, and saves everything for training
    """

    # Trial sequence: 2 classes, n_per_class each, randomly shuffled
    trial_sequence = build_trial_sequence(n_per_class, seed=run)
    
    # Buffers for the full session continuously 
    eeg = np.zeros((8, 0))
    aux = np.zeros((3, 0))
    timestamp = np.zeros((0))

    # stores extracted trials
    eeg_trials = []
    aux_trials = []
    events = []  # list of {"sample": int, "label": 0|1}

    
    # visual stimuli: instructions -> countdown -> recording -> trial #
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
            # eeg_in shape -> (8,25)
            eeg_in, aux_in, timestamp_in = queue_in.get()
            print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            aux = np.concatenate((aux, aux_in), axis=1) # continuous data ((8,100) -> (8,125))
            timestamp = np.concatenate((timestamp, timestamp_in), axis=0) # (100,) -> (125,)

        # Display trial number
        trial_stim.text = f"Trial {i_trial + 1} / {len(trial_sequence)}"

        # Preparation prompt
        instruction_stim.text = f"Prepare: {INSTRUCTION_TEXT[label]}" # ex: Prepare: Move LEFT HAND
        instruction_stim.draw() # queues instruction
        trial_stim.draw() # queues trial number
        window.flip() # displays both
        core.wait(1.5) # rest time for user

        # 3, 2, 1 Countdown with 0.5 seconds before next number
        for count in [3, 2, 1]:
            countdown_stim.text = str(count)
            instruction_stim.draw()
            countdown_stim.draw()
            trial_stim.draw()
            window.flip()
            core.wait(0.5)

        # Collect queued data right before recording onset (during prepare screen and countdown)
        while not queue_in.empty():
            eeg_in, aux_in, timestamp_in = queue_in.get() # get chunk
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
        baseline_duration_samples = int(baseline_duration * sampling_rate) # converts baseline duration from seconds to samples
        trial_duration_samples = int(record_duration * sampling_rate) # converts imagery duration to samples
        trial_start = max(0, cue_sample - baseline_duration_samples) # computes where extracted trial should start (some EEG before the queue)
        trial_end_required = trial_start + baseline_duration_samples + trial_duration_samples
        if trial_end_required > eeg.shape[1]:
            print(f'Warning: Not enough data for trial {i_trial}, skipping trial extraction')
            continue
        filtered_eeg = mne.filter.filter_data(eeg, sfreq=sampling_rate, l_freq=8, h_freq=30, verbose=False) # bandpass filter between 8 and 30Hz
        trial_eeg = np.copy(filtered_eeg[:, trial_start:trial_end_required]) # single eeg trial for 8 channels
        trial_aux = np.copy(aux[:, trial_start:trial_end_required])  # single aux trial
        print(f'trial {i_trial}: ', trial_eeg.shape, trial_aux.shape)
        baseline_average = np.mean(trial_eeg[:, :baseline_duration_samples], axis=1, keepdims=True) # mean EEG value during baseline for each channel -> (8,1)
        trial_eeg -= baseline_average
        eeg_trials.append(trial_eeg)
        aux_trials.append(trial_aux)
        events.append({"sample": int(cue_sample), "label": int(label)})

        # save data collected so far and quite program if Esc pressed
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

    # Final collection after the last trial but before program exits
    while not queue_in.empty():
        eeg_in, aux_in, timestamp_in = queue_in.get()
        print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
        eeg = np.concatenate((eeg, eeg_in), axis=1)
        aux = np.concatenate((aux, aux_in), axis=1)
        timestamp = np.concatenate((timestamp, timestamp_in), axis=0)

    if True:
        # save data
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


def _preprocess_chunk(chunk, sfreq):

    x = mne.filter.filter_data(
        chunk,
        sfreq=sfreq,
        l_freq=8, # 8-30 for mu + beta imagery band for CSP-based motor imagery decoding
        h_freq=30,
        verbose=False
    )

    x = x - np.mean(x, axis=1, keepdims=True) # centers signal around zero

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

    # Coordinate system -> bottom left is (0,0) and (col, row)
    _UP = (0, 1)
    _DOWN = (0, -1)
    _LEFT = (-1, 0)
    _RIGHT = (1, 0)

    # grid_n: grid is 20x20
    # cell_px: 30 pixels wide/tall
    # move_interval: snake moves once every 0.35 seconds
    def __init__(self, psychopy_window, grid_n=20, cell_px=30, move_interval=0.35):
        self.win = psychopy_window
        self.grid_n = grid_n
        self.cell_px = cell_px
        self.move_interval = move_interval

        ## Turn lookup tables (take perspective of snake direction when turning)

        # Given current direction, what direction a LEFT turn produces
        # ex: If the current direction is _LEFT, a LEFT turn leads to direction being _DOWN
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

        # background for the game
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
    # Put snake game back to starting state
    def reset(self):
        mid = self.grid_n // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)] # [head, body, tail]
        self._snake_set = set(self.snake) # set of snake coordinates

        self._heading = self._RIGHT
        self.food = self._place_food()

        self.score = 0
        self.alive = True

        self._pending = None
        self._waiting_for_turn = False
        self._last_move_t = core.getTime()

    # ==========================================================
    # External Control API for EEG
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
            return True # keep game alive but don't move yet
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

            dx, dy = self._heading # current movement direction

            # Vector from head to apple
            vec_x = apple_x - head_x
            vec_y = apple_y - head_y

            # Check if currently heading toward apple
            # If true, snake should keep going. If false, pause and ask for a turn decision
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

        # always grow by 1
        self.snake.insert(0, new_head)
        self._snake_set.add(new_head)

        # Eat apple
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.score_txt.text = f'Score: {self.score}'
        else:
            # if apple wasn't eaten we 
            removed = self.snake.pop()
            self._snake_set.discard(removed) # remove raises KeyError, so using discard to prevent crashes incase of inconsistencies

        return True

    # ==========================================================
    # Draw
    # ==========================================================
    def draw(self):
        """Render the game. (border, apple, snake, score)"""
        ox = -self.grid_n * self.cell_px / 2 # grid (0,0) -> pixel (-300, -300)
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
                random.randint(0, self.grid_n - 1), # inclusive of 0 and self.grid_n-1
                random.randint(0, self.grid_n - 1)
            )
            if pos not in self._snake_set:
                return pos

def run_snake_game():
    """
    Connects: EEG stream -> preprocessig -> CSP -> LDA -> left/right decision -> snake turn
    """
    global model, csp # reference model and csp defined earlier

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

    n_ch = 8 # 8 eeg channels
    win_samp = int(realtime_window_sec * sampling_rate) # every classification uses the latest (1.5 * 250) samples
    eeg_buf = np.zeros((n_ch, 0))

    # LIVE DATA
    while game.alive:

        ## Drain EEG queue
        while not queue_in.empty():
            eeg_in, _, _ = queue_in.get()
            eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)

            # keeps the most recent 750 samples
            if eeg_buf.shape[1] > 2 * win_samp:
                eeg_buf = eeg_buf[:, -2 * win_samp:]

        ## DECISION PHASE
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
            pred_hist = [] # list of predictions made during decision window

            while core.getTime() - decision_start < 1.5:
                
                # drain queue
                while not queue_in.empty():
                    eeg_in, _, _ = queue_in.get()
                    eeg_buf = np.concatenate((eeg_buf, eeg_in), axis=1)

                if eeg_buf.shape[1] >= win_samp: # check if buffer is large enough to classify
                    chunk = eeg_buf[:, -win_samp:]
                    ep = _preprocess_chunk(chunk, sampling_rate)[np.newaxis, ...] # turns (8, 375) -> (1,8,375)
                                                                                  # 1 EEG epoch, 8 channels, 375 samples
                    
                    # CSP expects (n_trials, n_channels, n_samples)
                    try:
                        feat = csp.transform(ep) # feat: (1,n) (each number is log variance of each filtered signal)
                        pred = model.predict(feat)[0]
                        pred_hist.append(pred)
                    except:
                        pass

                game.draw()
                window.flip()

            # Smooth prediction by averaging predictions
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
        text="Keyboard Test Mode\n<- = LEFT   -> = RIGHT\nESC = quit",
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
        keys = event.getKeys() # using event here because it polls the kerboard every frame and returns key presses
        
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
        # run_snake_game()

        run_snake_game_keyboard()
