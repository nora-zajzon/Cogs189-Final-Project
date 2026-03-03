# Saved eeg shape (8, 91986), 60 events to data/motor_imagery_2class/sub-01/ses-01/
 # Exception in thread Thread-3 (get_data):
 # Traceback (most recent call last):
 # File "C:\Users\tfei\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1009, in _bootstrap_inner
 # self.run()
 # File "C:\Users\tfei\AppData\Local\Programs\Python\Python310\lib\threading.py", line 946, in run
 # self._target(*self._args, **self._kwargs)
 # File "C:\Users\tfei\Downloads\Cogs189-Final-Project\run_mi.py", line 124, in get_data
 # data_in = board.get_board_data()
 # File "C:\Users\tfei\pyenv\lib\site-packages\brainflow\board_shim.py", line 1369, in get_board_data
 # data_size = self.get_board_data_count(preset)
 # File "C:\Users\tfei\pyenv\lib\site-packages\brainflow\board_shim.py", line 1318, in get_board_data_count
 # raise BrainFlowError('unable to obtain buffer size', res)
 # brainflow.exit_codes.BrainFlowError: BOARD_NOT_CREATED_ERROR:15 unable to obtain buffer size
 # 2.5790 WARNING Monitor specification not found. Creating a temporary one...
 # 385.7855 WARNING Stopping key buffers but this could be dangerous ifother keyboards rely on the same.
 # Exception ignored in: <function BoardShim.__del__ at 0x000002C78231E8C0>
 # Traceback (most recent call last):
 # File "C:\Users\tfei\pyenv\lib\site-packages\brainflow\board_shim.py", line 586, in __del__
 # File "C:\Users\tfei\pyenv\lib\site-packages\brainflow\board_shim.py", line 1353, in is_prepared
 # ctypes.ArgumentError: argument 1: <class 'ImportError'>: sys.meta_path is None, Python is likely shutting down
 
 # (pyenv) PS C:\Users\tfei\Downloads\Cogs189-Final-Project> python run_mi.py
 # {'accel_channels': [9, 10, 11], 'analog_channels': [19, 20, 21], 'ecg_channels': [1, 2, 3, 4, 5, 6, 7, 8], 'eeg_channels': [1, 2, 3, 4, 5, 6, 7, 8], 'eeg_names': 'Fp1,Fp2,C3,C4,P7,P8,O1,O2', 'emg_channels': [1, 2, 3, 4, 5, 6, 7, 8], 'eog_channels': [1, 2, 3, 4, 5, 6, 7, 8], 'marker_channel': 23, 'name': 'Cyton', 'num_rows': 24, 'other_channels': [12, 13, 14, 15, 16, 17, 18], 'package_num_channel': 0, 'sampling_rate': 250, 'timestamp_channel': 22}
 # [2026-02-20 15:31:26.495] [board_logger] [info] incoming json: {
 # "file": "",
 # "file_anc": "",
 # "file_aux": "",
 # "ip_address": "",
 # "ip_address_anc": "",
 # "ip_address_aux": "",
 # "ip_port": 0,
 # "ip_port_anc": 0,
 # "ip_port_aux": 0,
 # "ip_protocol": 0,
 # "mac_address": "",
 # "master_board": -100,
 # "other_info": "",
 # "serial_number": "",
 # "serial_port": "COM4",
 # "timeout": 0
 # }
 # [2026-02-20 15:31:26.495] [board_logger] [info] opening port \\.\COM4
 # Success: default$$$
 # Success: default$$$
 # Success: analog$$$
 # No model found. Run training first (scripts/train_motor.py).
 # 2.2348 WARNING Monitor specification not found. Creating a temporary one...
 # 16.8512 WARNING Stopping key buffers but this could be dangerous ifother keyboards rely on the same.


 1. fix the errors
 2. Decrease the amount of time during the movement measuring. They currently are prompted to move too long. 