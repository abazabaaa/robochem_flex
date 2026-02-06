"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Description:
Backend Class for the AVAPP spectrometer APP. This class handles the interaction
with the AVAlanCHE spectrometer, managing data acquisition, data polling, and
communication with the frontend GUI.
"""

# I swear 99 % of the time spent as a python dev is spent fixing FUCKING RELATIVE IMPORTS!
import sys
import os

dir_path = os.getcwd()
sys.path.append(dir_path)
"""------------------------------------------------------------------------------------------------------------------"""
from AVAlanCHE.AVAlanCHE_class import AVAlanCHE

import threading
import queue
import time

# mocking stays here for debug purposes, will be deleted at release for cleanliness
# from unittest.mock import MagicMock, PropertyMock
#
# magic_mock_object = MagicMock()
# magic_mock_object.side_effect = lambda *args, **kwargs: None
# magic_mock_object.run_measurement.side_effect = lambda: time.sleep(10)
import pandas as pd
import random


def ret_data():
    df = pd.DataFrame(
        {
            "Rshift": range(200, 3500),
            str(random.random()): [random.random() for _ in range(200, 3500)],
        }
    )
    return df


# # Create a MagicMock object
# magic_mock_object = MagicMock()
#
# # Set the 'data' attribute of the magic_mock_object to be a PropertyMock
# type(magic_mock_object).data = PropertyMock(side_effect=ret_data)


class AVAPP_Backend:
    """
    Backend class to manage spectrometer operations and GUI updates.

    This class is responsible for controlling the AVAlanCHE spectrometer,
    initiating data acquisition, polling for new data, and updating the GUI
    with the latest data.

    :param spec: Instance of the AVAlanCHE spectrometer for data acquisition.
    :type spec: AVAlanCHE
    :param update_gui_callback: Callback function to update the GUI with new data.
    :type update_gui_callback: function
    :param data_queue: Queue for receiving parameter commands from the frontend.
    :type data_queue: queue.Queue
    :param running: Flag to indicate the operational state of the backend.
    :type running: bool
    :param acq_thread: Thread for running the data acquisition process.
    :type acq_thread: threading.Thread
    :param poll_thread: Thread for polling the spectrometer and updating the GUI.
    :type poll_thread: threading.Thread
    """

    def __init__(self, update_gui_callback, parameter_queue, root):
        """
        Initialize the Backend instance.

        :param update_gui_callback: Function to call for updating the GUI.
        :type update_gui_callback: function
        :param parameter_queue: Queue for receiving parameter commands from the frontend.
        :type parameter_queue: queue.Queue
        """
        try:
            self.spec = AVAlanCHE()  # Initialize the spectrometer
            # self.spec = magic_mock_object
            self.update_gui_callback = update_gui_callback
            self.data_queue = parameter_queue
            self.plotting_queue = queue.Queue()
            self.root = root
            self.running = True
            self.acq_thread = None  # Initialize acquisition thread
            self.acq_thread_event = threading.Event()  # For auto releasing of threads
            self.poll_thread = None  # Initialize polling thread
            self.poll_thread_event = threading.Event()  # For auto releasing of threads
            self.killer_thread = None
        except Exception as e:
            self.running = False
            raise e

    def gather_params(self):
        """
        Gather and set parameters from the queue for the spectrometer.

        Polls the parameter_queue for parameter commands and applies them to the spectrometer.
        This method is intended to be called before starting the data acquisition.
        """
        try:
            command = self.data_queue.get(timeout=1)
            if command["command"] == "set_parameters":
                parameters = command["parameters"]
                self.parameter_save = parameters["save_as"]
                self.spec.set_parameters(parameters)
        except queue.Empty:
            pass

    def start_acq(self):
        """
        Start the data acquisition and polling processes in separate threads.

        Initiates the spectrometer data acquisition and the data polling in their
        respective threads, allowing the main application to remain responsive.
        """
        self.check_queue()
        # print('started checking')
        if self.acq_thread is None or not self.acq_thread.is_alive():
            self.acq_thread_event.clear()  # making sure you don't early terminate
            self.acq_thread = threading.Thread(target=self._acq_process)
            self.acq_thread.start()
        if self.poll_thread is None or not self.poll_thread.is_alive():
            self.poll_thread_event.clear()  # making sure you don't early terminate
            self.poll_thread = threading.Thread(target=self._poll_process)
            self.poll_thread.start()
        if self.killer_thread is None or not self.killer_thread.is_alive():
            self.killer_thread = threading.Thread(
                target=self.release_threads, daemon=True
            )
            self.killer_thread.start()

    def check_queue(self):
        if self.plotting_queue.qsize() > 0:
            data = self.plotting_queue.get()
            if type(data) == str and data == "STOP":
                # print("stopping queue check")
                return
            self.update_gui_callback(data, self.parameter_save)

        self.root.after(50, self.check_queue)

    # def _final_update(self):
    #     try:
    #         data = self.plotting_queue.get_nowait()
    #         self.update_gui_callback(data, self.parameter_save)
    #         print("data checked")
    #     except Exception as e:
    #         print("no data")
    #         pass

    def release_threads(self):
        """
        Release the threads from their waiting state.
        """
        self.acq_thread_event.wait()
        self.poll_thread_event.wait()

        if self.acq_thread is not None:
            self.acq_thread.join()
            self.acq_thread = None
        if self.poll_thread is not None:
            self.poll_thread.join()
            self.poll_thread = None

    def _acq_process(self):
        """
        Handle the data acquisition process in a separate thread.

        Manages the spectrometer's data acquisition, ensuring that data is
        collected properly and that the spectrometer's state is managed correctly.
        """
        try:
            self.gather_params()
            self.spec.setup_measurement()
            self.spec.get_dark()  # Get dark current data
            # time.sleep(10)
            self.spec.run_measurement()  # Start measurement # finished the acq time to join the threads
        except Exception as e:
            self.acq_thread_event.set()
            raise e
        finally:
            self.acq_thread_event.set()

    def _poll_process(self):
        """
        Poll the spectrometer for data and update the GUI in a separate thread.

        Continuously checks for new data from the spectrometer and updates
        the GUI by calling the update_gui_callback with the latest data.
        """
        try:
            while not self.acq_thread_event.is_set():
                time.sleep(self.spec.spec_polltime)
                data = self.spec.data
                if data is not None:
                    # print("data acquired")
                    self.plotting_queue.put(data)
                    # print(self.plotting_queue.qsize())

        except Exception as e:
            self.running = False
            raise e

        finally:
            # get the last data and put it in the queue
            time.sleep(self.spec.spec_polltime)
            data = self.spec.data
            if data is not None:
                # print("data acquired")
                self.plotting_queue.put(data)

            self.plotting_queue.put("STOP")
            self.poll_thread_event.set()

    def stop_acq(self):
        """
        Stop the data acquisition and polling processes.

        Signals the acquisition and polling threads to stop and waits for them to terminate,
        ensuring a clean shutdown of all operations.
        """
        # print("stopping")
        self.spec.stop()  # Stop the spectrometer  # Signal threads to stop
        # Ensure the acquisition thread is stopped
        if self.acq_thread is not None:
            self.acq_thread.join()
            self.acq_thread = None
        # Ensure the polling thread is stopped
        if self.poll_thread is not None:
            self.poll_thread.join()
            self.poll_thread = None

    def save_now(self):
        """
        Save the current data from the spectrometer.

        Triggers the spectrometer to save the current data, ensuring that
        the data is not lost.
        """
        self.spec.save_data()

    def stop(self):
        """
        Stop the backend process.

        Sets the running flag to False, signaling all operations to terminate.
        """
        self.acq_thread_event.set()
        self.poll_thread_event.set()
        # Ensure all threads are killed
        if self.acq_thread is not None and self.acq_thread.is_alive():
            self.acq_thread.join()
            self.acq_thread = None
        if self.poll_thread is not None and self.poll_thread.is_alive():
            self.poll_thread.join()
            self.poll_thread = None
        if self.killer_thread is not None and self.killer_thread.is_alive():
            self.killer_thread.join()
            self.killer_thread = None
        # destroy the spec:
        self.spec.kill_process()
