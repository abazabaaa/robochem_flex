"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:
This is the script on server side that controls the spectrometer. is built on the avaspec.py dll from Avantes. Documentation
for which can be found on the group drive under: General/13.instrumentsData/AvantesDLL

The main idea is to have a class that the server can instantiate, then the class has modules for setup, firing, dark saving,
and plotting. The server can then call these modules as needed.

Heads up This only works if
1) the spectrometer is connected to the rasberry pi via USB
2) the raspberry pi has the avantes dll installed (otherwise the raspberry has no way of telling the spectrometer what to do)
3) the gods have decided that you are worthy of using the spectrometer

"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "AVAlanCHE"))

import avaspec as ava
import RPi.GPIO as GPIO
import numpy as np
from datetime import datetime
import pandas as pd
import threading
import time
from queue import Queue
from utils import setup_logger


def error_handler(jerry):
    def wrapper(*args, **kwargs):
        try:
            return jerry(*args, **kwargs)
        except Exception as e:
            print(f"Something went wrong: {e}")

    return wrapper


def wlen_to_shift(wlen: np.array, wlen0=785.0):
    """converts wavelength in nm to raman shift (cm^-1)
    :param wlen: wavelength in nm (np.array)
    :param wlen0: laser wavelength in nm
    :return: raman shift in cm^-1 (np.array)
    """
    return 1e7 * (1 / wlen0 - 1 / wlen)


class AVAlanCHE:
    def __init__(self):
        """
        Initializes the AVAlanCHE class, the primary interface for controlling the Raman spectrometer.

        Upon instantiation, this class initializes the spectrometer device, sets up the default measurement
        configuration, and prepares the system for taking measurements. It establishes logging, initializes
        the spectrometer library, sets up device and laser control, and defines default parameters for
        spectrometer operations.

        Attributes:
            logger (logging.Logger): Logger for the class, used to log informational messages and errors.
            ready_to_measure (bool): Flag indicating whether the spectrometer is ready to take measurements.
            default_parameters (dict): Dictionary containing default spectrometer operation parameters, such as
                                       integration time, number of averages, measurement delay, and save settings.
            current_parameters (dict): A copy of `default_parameters` that can be modified based on runtime
                                       configurations.
            stop_requested (threading.Event): An event flag used primarily in GUI applications to signal when
                                               measurement processes should be halted.

        Operations:
            - Initializes the spectrometer and sets it ready for measurements.
            - Sets up a default measurement configuration with predefined parameters.
            - Prepares the laser control for the spectrometer.
            - Establishes a default save path for data collected during spectrometer operations.

        Note:
            This class is designed to be simple and straightforward for interfacing with the Raman spectrometer
            without sacrificing performance. It encapsulates the necessary setup, configuration, and control
            functionalities required to operate the spectrometer effectively.
        """

        self.log = setup_logger(self.__class__.__name__, "AVAlanCHE.log")
        self.log.info("Initializing AVAlanCHE class...")

        try:
            # Initialize the spectrometer library
            self.log.debug("Initializing AvaSpec library...")
            ava.initialise()
            self.log.info("AvaSpec library initialized successfully.")
        except Exception as e:
            self.log.exception("Error initializing AvaSpec library: %s", e)
            raise

        self.ready_to_measure = False
        self.log.info("Set ready_to_measure to False.")

        try:
            # Set up the device
            self.log.debug("Setting up the spectrometer device...")
            self.setup_device()
            self.log.info("Spectrometer device setup successfully.")
        except Exception as e:
            self.log.exception("Error during device setup: %s", e)
            raise

        try:
            # Initiate measurement configuration
            self.log.debug("Initializing measurement configuration...")
            self.measuring = True
            self.init_measconfig()
            self.log.info("Measurement configuration initialized.")
        except Exception as e:
            self.log.exception("Error initializing measurement configuration: %s", e)
            raise

        # Set default parameters
        self.default_parameters = {
            "integration_time": 1000.0,
            "n_averages": 1,
            "n_scans": 1,
            "delay (ms)": 1000.0,
            "correct_dark": True,
            "dynamic_correct_dark": False,
            "save_as": "single",
            "save_path": self._default_save_path(),
            "safety_threshold": 10000,
            "save_moniker": None,
            "save": True,
        }
        self.log.debug(f"Default parameters set: {self.default_parameters}")

        self.current_parameters = self.default_parameters.copy()
        self.log.debug(
            "Current parameters initialized as a copy of default parameters."
        )

        try:
            # Setup laser control
            self.log.debug("Setting up laser control...")
            self.setup_laser_control()
            self.log.info("Laser control setup successfully.")
        except Exception as e:
            self.log.exception("Error during laser control setup: %s", e)
            raise

        # Calculate spectrometer polling time
        self.spec_polltime = (
            (
                self.current_parameters["integration_time"]
                * self.current_parameters["n_averages"]
            )
            + self.current_parameters["delay (ms)"]
        ) / 1000.0
        self.log.debug(
            f"Spectrometer polling time calculated: {self.spec_polltime} seconds"
        )

        # Initialize stop event for GUI apps
        self.stop_requested = threading.Event()
        self.log.info("Stop_requested event initialized.")

        self.log.info("AVAlanCHE class initialized successfully.")

    def _default_save_path(self):
        """
        Generates the default save path for storing Raman spectra data.

        This function constructs a path to save Raman spectra data based on the current date.
        It follows the structure '/mnt/Raman_Data/robochem_spectra/YYYYMMDD', where 'YYYYMMDD'
        represents the year, month, and day when the data is being saved. If the directory
        for the current date does not exist, it is created.

        Returns:
            str: The path to the directory where data should be saved for the current date.
        """
        self.log.info("Generating default save path for Raman spectra data.")
        try:
            # Define the base path
            path = os.path.join("/mnt", "Raman_Data", "raman_spectra")
            self.base_mounted_path = path
            self.log.debug(f"Base mounted path set to: {path}")

            # Construct the folder path for today's date
            today_folder = os.path.join(path, datetime.now().strftime("%Y%m%d"))
            self.log.debug(f"Today's folder path: {today_folder}")

            # Create the folder if it doesn't exist
            if not os.path.exists(today_folder):
                os.makedirs(today_folder)
                self.log.info(f"Created directory for today's spectra: {today_folder}")
            else:
                self.log.info(f"Directory already exists: {today_folder}")

            return today_folder
        except Exception as e:
            self.log.exception(f"Error generating default save path: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit function for the spectrometer control class.

        This method is called when exiting the runtime context of the class. It is responsible for
        cleaning up resources by closing the connection to the spectrometer. It performs necessary
        cleanup operations, such as releasing GPIO resources and finalizing the spectrometer's
        connection using the AVS_Done method.

        Args:
            exc_type (Exception or None): The type of exception that caused the exit, if any.
            exc_val (Exception or None): The exception instance that caused the exit, if any.
            exc_tb (traceback or None): The traceback information, if an exception occurred.

        Note:
            This method ensures that the resources are properly released regardless of whether
            the context was exited normally or due to an exception.
        """
        self.log.info("Exiting the AVAlanCHE class context manager.")
        GPIO.cleanup()
        ava.AVS_Done()

    @error_handler
    def __del__(self):
        """
        Destructor for the spectrometer control class.

        This method is invoked when an instance of the class is about to be destroyed. It ensures
        that the connection to the spectrometer is properly closed and that any associated resources,
        such as GPIO, are cleaned up. This is a safeguard to ensure that resources are not left
        allocated when the object is garbage collected.

        Note:
            It is generally recommended to explicitly manage resources with context managers or
            explicit close methods. Relying on __del__ for resource cleanup can be less reliable
            due to the unpredictability of garbage collection timing in Python.
        """
        self.log.info("Deleting the AVAlanCHE class instance.")
        GPIO.cleanup()
        ava.AVS_Done()

    @property
    def data(self):
        """
        Property that provides access to the newest data as a pandas DataFrame.

        This property acts as a convenient way to access the latest data captured by the spectrometer.
        It leverages the `poll_data` method to retrieve the data, ensuring that the caller always receives
        the most recent dataset available, formatted as a pandas DataFrame.

        Returns:
            pandas.DataFrame: The newest dataset available from the spectrometer, or None if no data is available.
        """
        self.log.info("Accessing the newest data as a pandas DataFrame.")
        return self.poll_data()

    @property
    def data_server(self):
        """
        Property of data polling for the server, access newest data as a pd DataFrame, if no data has been put in the
        queue returns "waiting". So the robochem knows if we have to keep polling or we are done.

        returns:
            pandas.DataFrame: The Newest dataset available, or
            str: waiting
        """
        self.log.info("Accessing data_server property.")
        try:
            # Poll data
            data = self.poll_data()
            self.log.debug(f"Data polled: {type(data).__name__}")

            # Check if the data is a DataFrame
            if isinstance(data, pd.DataFrame):
                self.log.info("New data received as a DataFrame.")
                return data

            # If no data but still measuring, return "waiting"
            if data is None and self.measuring:
                self.log.info("No data available yet, returning 'waiting'.")
                return "waiting"

            # Return other data types or end signals
            self.log.info(f"Data received: {data}")
            return data
        except Exception as e:
            self.log.exception(f"Error accessing data_server: {e}")
            raise

    def poll_data(self):
        """
        Polls the internal queue for the newest dataset available.

        This method checks the internal data queue for any available datasets. If data is present,
        it retrieves (dequeues) the newest data set from the queue. This mechanism ensures that data
        consumption is FIFO (First In, First Out), allowing for sequential processing of datasets as they
        are generated.

        Returns:
            pandas.DataFrame or None: The newest data set from the queue as a pandas DataFrame if available,
            otherwise None if the queue is empty.
        """
        self.log.info("Polling data from the internal queue.")
        try:
            # Check if the data queue is not empty
            if not self.data_queue.empty():
                self.log.debug("Data queue is not empty. Retrieving data.")
                data = self.data_queue.get_nowait()
                self.log.info("Data successfully retrieved from the queue.")
                return data
            else:
                self.log.info("Data queue is empty. No data to retrieve.")
                return None
        except Exception as e:
            self.log.exception(f"Error while polling data: {e}")
            raise

    def return_data(self) -> pd.DataFrame:
        """
        Transforms and returns the current spectrometer data as a pandas DataFrame.

        This method is responsible for transforming the raw data captured from the spectrometer into a structured
        pandas DataFrame. It also applies specific formatting based on the 'save_as' parameter, which dictates the
        structure of the returned DataFrame. The method ensures that only relevant columns are retained based on the
        'save_as' mode ('single', 'monitor', or 'kinetic').

        Returns:
            pandas.DataFrame: The current spectrometer data formatted as a DataFrame. The structure of the DataFrame
            varies based on the 'save_as' mode:
                - 'single' and 'monitor': Drops all columns except 'Rshift', then adds the new spectrum as a column.
                - 'kinetic': Simply adds the new spectrum as a new column to the DataFrame.
        """
        self.log.info(
            "Transforming and returning spectrometer data as a pandas DataFrame."
        )
        try:
            # Check if the DataFrame exists; if not, initialize it
            if not hasattr(self, "data_df"):
                self.log.debug("Initializing data DataFrame with 'Rshift' column.")
                self.data_df = pd.DataFrame({"Rshift": self.wavelength})

            save_as = self.current_parameters.get("save_as")
            self.log.debug(f"'save_as' parameter: {save_as}")

            if save_as in ["single", "monitor", "server_single"]:
                # For 'single' or 'monitor', reset and add the new column
                self.log.debug("Processing data in 'single' or 'monitor' mode.")
                self._reset_data_frame()  # Keep only the first column
                self.data_df[self.current_time] = self.current_spectrum
                self.log.info(
                    f"New spectrum added to DataFrame with timestamp: {self.current_time}."
                )

            elif save_as in ["kinetic", "server_kinetic"]:
                # For 'kinetic', just add the new column
                self.log.debug("Processing data in 'kinetic' mode.")
                self.data_df[self.current_time] = self.current_spectrum
                self.log.info(
                    f"New spectrum added to DataFrame with timestamp: {self.current_time}."
                )

            self.log.debug("Data transformation completed successfully.")
            return self.data_df

        except Exception as e:
            self.log.exception(f"Error in transforming spectrometer data: {e}")
            raise

    def setup_laser_control(self):
        """
        Initializes the GPIO settings for laser control.

        This method sets up the GPIO mode and configures the designated pin for TTL control of the laser.
        It specifies the pin used for controlling the laser's on/off state and sets it as an output pin.
        """
        self.log.info("Initializing GPIO settings for laser control.")
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            self.log.debug("GPIO mode set to BCM.")

            # Configure the TTL control pin
            self.pin_ttl = 17
            GPIO.setup(self.pin_ttl, GPIO.OUT)
            self.log.info(f"Laser control pin {self.pin_ttl} configured as OUTPUT.")
        except Exception as e:
            self.log.exception(f"Error in setting up laser control: {e}")
            raise

    def laser_on(self):
        """
        Turns the laser on by setting the designated GPIO pin high.

        This method activates the laser by sending a high signal (GPIO.HIGH) to the TTL control pin,
        effectively turning the laser on.
        """
        self.log.info("Turning the laser on.")
        GPIO.output(self.pin_ttl, GPIO.HIGH)

    def laser_off(self):
        """
        Turns the laser off by setting the designated GPIO pin low.

        This method deactivates the laser by sending a low signal (GPIO.LOW) to the TTL control pin,
        effectively turning the laser off.
        """
        self.log.info("Turning the laser off.")
        GPIO.output(self.pin_ttl, GPIO.LOW)

    def stop(self):
        """
        Stops the acquisition process and performs necessary cleanup.

        This method signals to stop the current acquisition process. It ensures that any final data
        is saved by calling `save_data` with `final_save=True`. Additionally, it sets a flag (`stop_requested`)
        to indicate that a stop has been requested, which can be checked by ongoing or future processes to
        gracefully terminate their operation.
        """
        self.log.info("Stopping the acquisition process.")
        self.save_data(final_save=True)
        self.stop_requested.set()

    def unset(self):
        """
        Clears the stop request flag.

        This method resets the `stop_requested` flag by clearing it, indicating that stopping the acquisition
        process is no longer requested. This allows for the resumption or initiation of new acquisition processes
        without the impediment of a previously set stop request.
        """
        self.log.info("Clearing the stop request flag.")
        self.stop_requested.clear()

    @error_handler
    def setup_device(self):
        """
        Initializes and activates the spectrometer device.

        This method performs the initial setup for the spectrometer device, including initialization with
        AVS_Init, retrieval of connected devices with AVS_GetList, and activation of the first detected
        spectrometer with AVS_Activate. It also initializes a data queue for storing and returning data to
        the server or GUI.

        Raises:
            Exception: If no spectrometer devices are connected or if the spectrometer cannot be activated.

        Note:
            This method is decorated with @error_handler, which implies that it has error handling
            mechanisms to manage exceptions or errors during the device setup process.
        """
        self.log.info("Initializing spectrometer connection.")

        try:
            # Initialize the spectrometer and check the number of devices connected
            self.n_devices = ava.AVS_Init(0)
            self.log.info(f"Number of devices connected: {self.n_devices}")

            if self.n_devices <= 0:
                self.log.error("No spectrometer connected.")
                raise Exception("No spectrometer connected")

            # Get the list of available devices
            self.devices = ava.AVS_GetList()
            if not self.devices:
                self.log.error("No spectrometer devices found.")
                raise Exception("No spectrometer connected")

            # Activate the first available spectrometer
            self.spectrometer = ava.AVS_Activate(self.devices[0])
            if self.spectrometer == -1:
                self.log.error("Failed to activate spectrometer.")
                raise Exception("Could not activate spectrometer")

            # Initialize the data queue
            self.data_queue = Queue()
            self.log.info("Data queue initialized successfully.")

        except Exception as e:
            self.log.exception(f"Error during spectrometer initialization: {e}")
            raise

    def init_measconfig(self):
        """
        Initializes the measurement configuration with default parameters.

        This method sets up a default measurement configuration for the spectrometer. The configuration
        includes settings such as the start and stop pixels, integration time, number of averages,
        dynamic dark correction settings, smoothing parameters, saturation detection, and trigger
        settings. These parameters are set to their default values, suitable for a standard measurement
        setup.

        Note:
            This method sets the default for the spectrometer's measurement. To set custom parameters use the
            `set_parameters` method.
        """
        self.meas_config = ava.MeasConfigType()
        self.meas_config.m_StartPixel = 0
        self.meas_config.m_StopPixel = 2047
        self.meas_config.m_IntegrationTime = 1000.0
        self.meas_config.m_IntegrationDelay = 0
        self.meas_config.m_NrAverages = 1
        self.meas_config.m_CorDynDark_m_Enable = 1
        self.meas_config.m_CorDynDark_m_ForgetPercentage = 0
        self.meas_config.m_Smoothing_m_SmoothPix = 0
        self.meas_config.m_Smoothing_m_SmoothModel = 1
        self.meas_config.m_SaturationDetection = 0
        self.meas_config.m_Trigger_m_Mode = 0
        self.meas_config.m_Trigger_m_Source = 0
        self.meas_config.m_Trigger_m_SourceType = 0
        self.meas_config.m_Control_m_StrobeControl = 0
        self.meas_config.m_Control_m_LaserDelay = 0
        self.meas_config.m_Control_m_LaserWidth = 0
        self.meas_config.m_Control_m_LaserWaveLength = 0.0
        self.meas_config.m_Control_m_StoreToRam = 0

    def set_parameters(self, parameters: dict) -> None:
        """
        Sets the spectrometer configuration parameters based on input from the server or the GUI.

        This method takes a dictionary of parameters specifying how the spectrometer should
        operate, including integration time, number of averages, number of scans, saving frequency,
        dark correction, and saving behavior. Some parameters are directly applied to the
        spectrometer's measurement configuration object, while others are stored as class attributes.

        Parameters:
            parameters (dict): A dictionary containing the spectrometer operational parameters.
                - integration_time (float): The integration time for each measurement in milliseconds.
                - n_averages (int): The number of averages per scan to reduce noise.
                - n_scans (int): The total number of scans (spectra) to be taken.
                - delay (float): Delay between each scan in ms (used when n_scans > 1).
                - correct_dark (bool): Specifies whether to perform dark correction before measurements.
                - save_as (string): Determines the saving behavior ('single', 'kinetic', 'monitor').

        Raises:
            Exception: If a required parameter like 'save_path' is not specified in the input parameters.

        Usage:
            - For a single spectrum, set n_scans=1.
            - To take and save 100 spectra as quickly as possible, set n_scans=100 and delay = 0.
            - To take 10 spectra with a delay of 10s between each, set n_scans=10 and delay = 10000

        Note:
            The method validates the provided parameters against the expected keys and updates the
            spectrometer's measurement configuration accordingly. Unrecognized parameters are logged
            but do not halt execution. Critical parameters like 'integration_time' and 'n_averages'
            are directly applied to the measurement configuration.
        """

        self.log.info("Setting spectrometer configuration parameters.")
        try:
            # Update current parameters based on the input dictionary
            for key, value in parameters.items():
                if key in self.current_parameters.keys():
                    if key == "save_path":
                        value = self._fix_path(value)
                    self.current_parameters[key] = value
                    self.log.debug(f"Parameter updated: {key} = {value}")
                else:
                    self.log.warning(f"Unrecognized parameter: {key}")

            # Apply key parameters to the measurement configuration object
            self.meas_config.m_IntegrationTime = self.current_parameters[
                "integration_time"
            ]
            self.log.debug(
                f"Integration time set to: {self.current_parameters['integration_time']} ms"
            )
            self.meas_config.m_NrAverages = self.current_parameters["n_averages"]
            self.log.debug(
                f"Number of averages set to: {self.current_parameters['n_averages']}"
            )

            # Calculate and update spectrometer polling time
            self.spec_polltime = (
                (
                    self.current_parameters["integration_time"]
                    * self.current_parameters["n_averages"]
                )
                + self.current_parameters["delay (ms)"]
            ) / 1000.0
            self.log.debug(
                f"Spectrometer poll time calculated: {self.spec_polltime} seconds"
            )

            # Validate critical parameters
            if self.current_parameters["save_path"] is None:
                self.log.error("Save path is not specified.")
                raise Exception("No save path specified, please specify a save path")

            self.log.info("Spectrometer parameters set successfully.")

        except Exception as e:
            self.log.exception(f"Error while setting parameters: {e}")
            raise

    def _fix_path(self, path: str):
        """
        Fixes the path to sace the spectra properly

        - if the path does not start with '/mnt/Raman_Data/' prepend self.base_mounted_path.
        - converts windows-style backslashes to the superior Unix-style forward slashes
        - Ensures the directory exists

        :param path:str the path to fix
        :returns str: the fixed path
        """
        self.log.info("Fixing the provided path for spectra saving.")
        try:
            # Replace Windows-style backslashes with Unix-style forward slashes
            fixed_path = path.replace("\\", "/")
            self.log.debug(f"Replaced backslashes: {fixed_path}")

            # Prepend base path if it does not start with the correct prefix
            if not fixed_path.startswith("/mnt/Raman_Data/"):
                self.log.debug(
                    "Path does not start with '/mnt/Raman_Data/'. Adjusting path..."
                )
                if fixed_path.startswith("/"):
                    fixed_path = fixed_path.lstrip("/")
                fixed_path = os.path.join(self.base_mounted_path, fixed_path)
                fixed_path = os.path.normpath(fixed_path)
                self.log.debug(f"Path adjusted to: {fixed_path}")

            # Ensure the directory exists
            os.makedirs(fixed_path, exist_ok=True)
            self.log.info(f"Directory ensured: {fixed_path}")

            return fixed_path

        except Exception as e:
            self.log.exception(f"Error fixing path: {path}")
            raise ValueError(
                f"Tried to create the directory {fixed_path} but encountered an error: {str(e)}"
            )

    @error_handler
    def take_spectrum(self) -> None:
        """
        Initiates the spectrometer to take a single spectrum and updates the current spectrum and wavelength data.

        This method triggers the spectrometer to perform a measurement based on previously set parameters. It
        checks if the system is ready for measurement, initiates the spectrum acquisition, and waits for the
        acquisition to complete. The method also ensures that the wavelength data is initialized before
        fetching the latest spectrum data.

        The spectrum acquisition process involves starting a measurement with a predefined number of scans
        (in this case, 1 scan), polling the spectrometer for completion, and then retrieving the spectrum data
        along with the timestamp of acquisition. If the wavelength data has not been set up prior to this call,
        it initializes the wavelength data based on the spectrometer's response.

        Raises:
            Warning: Logs and prints a warning message if the spectrometer is not ready to measure, indicating
                     that the necessary setup procedures might not have been completed successfully.

        Returns:
            np.array: The acquired spectrum as a NumPy array, trimmed to the first 2047 pixels. This return
                      statement is implied by the description and usage but not explicitly included in the
                      method signature as the method's return type is None.

        Note:
            The readiness to measure is determined by a flag (`ready_to_measure`) that should be set true
            by preceding setup or configuration methods. The polling interval for the scan completion check
            is determined by `spec_polltime`, which should be configured to suit the measurement setup.

        Usage:
            This method is intended to be called when the spectrometer is fully configured and ready to start
            taking spectra. It is a blocking call that only returns after a spectrum has been successfully
            acquired or if the spectrometer is not ready for measurement.
        """
        if not self.ready_to_measure:
            self.log.warning("Not ready to measure. Measurement cannot proceed.")
            print("Not ready to measure")
            return

        try:
            # Start the measurement (specify the correct number of scans)
            self.log.info("Starting measurement with 1 scan.")
            ava.AVS_Measure(self.spectrometer, 0, 1)

            # Loop until the spectrometer signals that it is done
            self.log.debug("Polling spectrometer for completion.")
            while not ava.AVS_PollScan(self.spectrometer):
                time.sleep(0.05)

            self.log.info("Measurement completed successfully.")

            # Check if we already have wavelength information
            if not hasattr(self, "wavelength"):
                self.log.debug(
                    "Retrieving wavelength data and converting to Raman shift."
                )
                self.wavelength = wlen_to_shift(
                    np.array(ava.AVS_GetLambda(self.spectrometer)[:2047])
                )
                self.log.info("Wavelength data converted to Raman shift.")

            # Get the spectrum data
            self.log.debug("Retrieving spectrum data from spectrometer.")
            tstamp, spectrum = ava.AVS_GetScopeData(self.spectrometer)
            self.log.info("Spectrum data retrieved successfully.")
            return np.array(spectrum[:2047])  # Cut the spectrum to 2047 pixels

        except Exception as e:
            self.log.exception(f"Error during measurement: {e}")
            raise

    @error_handler
    def get_dark(self) -> None:
        """
        Acquires a dark/reference spectrum for subsequent measurements.

        This method is intended to be run during the setup phase to capture a dark spectrum
        with the laser turned off. It temporarily overrides the number of averages to 3 for
        increased accuracy in the dark spectrum measurement, captures the dark spectrum,
        and then restores the original number of averages. The captured dark spectrum is
        set as the reference for correcting future measurements.

        Note:
            This method adjusts `n_averages` in `current_parameters` to ensure the dark
            spectrum is averaged over 3 measurements for improved stability and accuracy.
            It automatically restores the original `n_averages` value after the dark
            spectrum is acquired.
        """
        self.log.info("Starting process to acquire a dark spectrum.")
        try:
            # Turn off the laser
            self.log.info("Turning off the laser to capture the dark spectrum.")
            self.laser_off()

            # Save the original number of averages and override it to 3
            original_n_averages = self.current_parameters["n_averages"]
            self.log.debug(f"Original n_averages: {original_n_averages}")
            self.current_parameters["n_averages"] = 3
            self.log.debug(
                "Overridden n_averages set to 3 for dark spectrum acquisition."
            )

            # Apply updated parameters and setup the measurement
            self.set_parameters(self.current_parameters)
            self.setup_measurement()
            self.log.info("Measurement setup updated for dark spectrum acquisition.")

            # Capture the dark spectrum
            self.log.info("Capturing dark spectrum.")
            self.current_dark = self.take_spectrum()
            self.dark_timestamp = time.time()
            self.log.info(
                f"Dark spectrum captured successfully at timestamp: {self.dark_timestamp}"
            )

            # Restore the original number of averages
            self.current_parameters["n_averages"] = original_n_averages
            self.set_parameters(self.current_parameters)
            self.setup_measurement()
            self.log.info(f"Restored original n_averages: {original_n_averages}")

            # Turn the laser back on
            self.log.info("Turning the laser back on.")
            self.laser_on()

            # Countdown before resuming measurements
            self.log.info("Measurement will resume shortly.")
            for i in range(3, 0, -1):
                self.log.debug(f"Resuming in {i} seconds.")
                print(f"{i}...")
                time.sleep(1)
            self.log.info("Resuming measurements now.")
        except Exception as e:
            self.log.exception(f"Error during dark spectrum acquisition: {e}")
            raise

    @error_handler
    def setup_measurement(self) -> None:
        """
        Prepares the spectrometer for measurement.

        This method configures the spectrometer with the measurement settings specified in
        `meas_config` and sets a flag indicating that the spectrometer is ready to measure.
        It must be called before starting any measurement to ensure the spectrometer is
        correctly prepared and to enable the `ready_to_measure` flag.
        """
        self.log.info("Setting up the spectrometer for measurement.")
        ava.AVS_PrepareMeasure(self.spectrometer, self.meas_config)
        self.log.info("Spectrometer setup completed successfully.")
        self.ready_to_measure = True

    def run_measurement(self) -> None:
        """
        Conducts the measurement loop, capturing spectra and managing data accordingly.

        This method orchestrates the measurement process: it turns on the laser, checks if
        the system is ready to measure, and enters a loop to continuously take spectra.
        It applies dark correction if necessary, saves spectra based on the defined criteria,
        and ensures the loop exits safely after completing the desired number of measurements.
        The laser is turned off and final data is saved at the end of the measurement process.

        During the loop, it also manages a safety counter to prevent infinite loops, applies
        a delay between measurements as specified, and clears the data queue at the start to
        ensure fresh data collection. The method leverages several helper functions to
        streamline the process of taking spectra, applying corrections, and saving data.
        """
        # turn on laser:
        self.log.info("Starting measurement process.")

        try:
            # Turn on the laser
            self.log.info("Turning on the laser.")
            self.laser_on()

            # Validate readiness to measure
            self.validate_ready_to_measure()
            self.log.info("System validated and ready to measure.")

            # Clear the data queue if it's not empty
            if not self.data_queue.empty():
                self.log.debug("Data queue is not empty. Clearing the queue.")
                self.data_queue = Queue()

            save_counter = 0
            safety_counter = 0
            save_flag = False
            self.first_spectrum = True
            self.measuring = True

            self.log.info("Measurement parameters:")
            self.log.debug(self.current_parameters)

            # Start the measurement loop
            while self.should_continue_measurement(save_counter):
                safety_counter += 1
                save_counter += 1

                # Ensure safety by checking the safety counter
                self.log.debug(
                    f"Safety counter: {safety_counter}, Save counter: {save_counter}"
                )
                self.ensure_safety(safety_counter)

                # Take the spectrum and apply dark correction if necessary
                self.current_spectrum = self.take_spectrum()
                self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                self.log.info(f"Spectrum captured at {self.current_time}.")

                # Update dark spectrum if required
                self.dark_update()
                self.log.debug("Dark spectrum updated if necessary.")

                # Save the spectrum if needed
                self.save_spectrum_if_needed(save_flag)
                self.log.debug(f"Spectrum saved if needed. Save flag: {save_flag}")

                # Delay between measurements
                delay = self.current_parameters["delay (ms)"] / 1000.0
                self.log.debug(f"Delaying next measurement by {delay} seconds.")
                if self.should_continue_measurement(save_counter):
                    time.sleep(delay)

            self.measuring = False
            self.log.info("Measurement loop completed.")

            # Turn off the laser
            self.log.info("Turning off the laser.")
            self.laser_off()

            # Save final data
            self.log.info("Saving final data.")
            self.save_data(final_save=True)

            self.log.info("Measurement process completed successfully.")

        except Exception as e:
            self.log.exception(f"Error during the measurement process: {e}")
            raise

    def check_stats(self):
        """So I measured some spectra, and figured out that 1) the last 100 pixels very rarely have any bands in them
        apparently with a fresh dark the STD and Mean of the last 100 pixels are very similar (which is nice, it's shot noise)
        as the dark gets older both the mean and std increase but the mean increases faster, a good way to check if the dark
        is too old is to check if the mean of the last 100 pixels is within +- 3 STD, if it is the dark is still good.

        """
        self.log.info("Checking the validity of the dark spectrum.")

        try:
            # Calculate current stats for the last 100 pixels
            current_std = np.std(self.current_spectrum[-100:])
            current_mean = np.mean(self.current_spectrum[-100:])
            self.log.debug(f"Current STD: {current_std}, Current Mean: {current_mean}")

            # Check 1: Are both the mean and std larger than those saved?
            std_increased = np.abs(current_std) > np.abs(self.stats["std"])
            mean_increased = np.abs(current_mean) > np.abs(self.stats["mean"])
            self.log.debug(
                f"STD Increased: {std_increased}, Mean Increased: {mean_increased}"
            )

            # Calculate old and new ratios of mean to std
            old_ratio = (
                np.abs(self.stats["mean"] / self.stats["std"])
                if self.stats["std"] != 0
                else float("inf")
            )
            new_ratio = (
                np.abs(current_mean / current_std) if current_std != 0 else float("inf")
            )
            self.log.debug(f"Old Ratio: {old_ratio}, New Ratio: {new_ratio}")

            # Check 2: Has the ratio increased?
            ratio_increased = new_ratio > old_ratio
            self.log.debug(f"Ratio Increased: {ratio_increased}")

            # Is the new mean within 3 std of the old mean?
            mean_within_3std = (
                np.abs(current_mean - self.stats["mean"]) < 3 * current_std
            )
            self.log.debug(f"Mean within 3 STD: {mean_within_3std}")

            # Return True if all conditions are met
            result = (
                std_increased
                and mean_increased
                and ratio_increased
                and mean_within_3std
            )
            self.log.info(f"Dark spectrum validity check result: {result}")
            return result

        except Exception as e:
            self.log.exception(f"Error during dark spectrum validity check: {e}")
            raise

    def dark_update(self):
        """we noticed that the dask spectrum goes to shit after a while, this checks if the dark is:
        1) too old,
        2) if we subtract the dark from the spectrum and the "cleaned" spectrum has a lot of rubbish in the last 50 pixels
        """
        self.log.info("Updating dark spectrum if necessary.")

        try:
            # Skip updates if dynamic dark correction is disabled or if it's the first spectrum
            if (
                self.current_parameters["dynamic_correct_dark"] is False
                or self.first_spectrum
            ):
                self.log.info(
                    "Dynamic dark correction is disabled or this is the first spectrum. Applying dark correction."
                )
                self.apply_dark_correction_if_needed()
                return

            # 1) Check if the dark spectrum is older than 15 minutes
            time_elapsed = time.time() - self.dark_timestamp
            if time_elapsed > 15 * 60:
                self.log.warning(
                    f"Dark spectrum is too old (elapsed time: {time_elapsed / 60:.2f} minutes). Capturing a new dark spectrum."
                )
                self.get_dark()
                self.first_spectrum = True

            # 2) Check the "rubbish" in the last 50 pixels of the cleaned spectrum
            cleaned_max = max(self.current_spectrum[-50:] - self.current_dark[-50:])
            self.log.debug(
                f"Max value of cleaned spectrum in the last 50 pixels: {cleaned_max}"
            )
            if cleaned_max >= 150:
                self.log.warning(
                    "Dark spectrum correction results in excessive noise. Capturing a new dark spectrum."
                )
                self.get_dark()
                self.first_spectrum = True

            # Apply dark correction if needed
            self.log.info("Applying dark correction to the spectrum.")
            self.apply_dark_correction_if_needed()

        except Exception as e:
            self.log.exception(f"Error during dark spectrum update: {e}")
            raise

    def validate_ready_to_measure(self):
        """
        Validates if the spectrometer is ready for measurement and resets the stop request if set.

        This method checks if the spectrometer is prepared and ready to measure by verifying the
        `ready_to_measure` flag. It raises an exception if the system is not ready. Additionally,
        if a stop has been requested (indicated by the `stop_requested` flag being set), it resets
        this flag to allow measurements to proceed.

        Raises:
            Exception: If the spectrometer is not ready to measure or if other pre-measurement
                       checks fail, indicating the system is not properly configured or in a
                       state to start measurements.
        """
        self.log.info("Validating if the spectrometer is ready for measurement.")

        try:
            # Check if the spectrometer is ready
            if not self.ready_to_measure:
                self.log.error("Spectrometer is not ready to measure.")
                raise Exception("Not ready to measure")

            # Check if a stop request is set
            if self.stop_requested.is_set():
                self.log.warning(
                    "Stop request detected. Resetting stop request to proceed."
                )
                self.unset()
            self.log.info("Validation successful. Spectrometer is ready to measure.")

        except Exception as e:
            self.log.exception(f"Error during validation: {e}")
            raise

    def should_continue_measurement(self, save_counter) -> bool:
        """
        Determines whether to continue the measurement loop based on the number of scans.

        Parameters:
            save_counter (int): The current count of saved spectra.

        Returns:
            bool: True if the measurement process should continue (i.e., the number of completed
                  scans is less than the total number of scans requested and no stop has been
                  requested); False otherwise.
        """
        return (
            save_counter < self.current_parameters["n_scans"]
            and not self.stop_requested.is_set()
        )

    def ensure_safety(self, safety_counter):
        """
        Prevents the measurement loop from running indefinitely by enforcing a safety threshold.

        Parameters:
            safety_counter (int): A counter incremented with each iteration of the measurement loop.

        Raises:
            Exception: If the safety counter exceeds a predefined safety threshold, indicating
                       potential issues with the spectrometer's responsiveness or measurement loop
                       logic.
        """
        if safety_counter > self.current_parameters["safety_threshold"]:
            self.log.error("Safety threshold exceeded, spectrometer not responding")
            raise Exception("Safety threshold exceeded, spectrometer not responding")

    def apply_dark_correction_if_needed(self):
        """
        Applies dark correction to the current spectrum if enabled in the measurement parameters.

        This method subtracts the dark/reference spectrum from the current spectrum to correct for
        background noise and signal, assuming dark correction has been enabled in the configuration
        parameters (`correct_dark` is True).
        """
        if self.current_parameters["correct_dark"]:
            self.log.info("Applying dark correction to the current spectrum.")
            try:
                self.current_spectrum -= self.current_dark
                self.log.debug("Dark correction applied successfully.")
            except Exception as e:
                self.log.exception(f"Error applying dark correction: {e}")
                raise

        if self.first_spectrum:
            self.log.info(
                "Processing the first spectrum. Calculating initial statistics."
            )
            try:
                self.first_spectrum = False
                self.stats = {
                    "std": np.std(self.current_spectrum[-100:]),
                    "mean": np.mean(self.current_spectrum[-100:]),
                }
                self.log.debug(
                    f"Initial statistics calculated: std={self.stats['std']}, mean={self.stats['mean']}."
                )
            except Exception as e:
                self.log.exception(
                    f"Error calculating initial statistics for the first spectrum: {e}"
                )
                raise

    def save_spectrum_if_needed(self, save_flag):
        """
        Determines the conditions under which the current spectrum should be saved and executes saving.

        Parameters:
            save_flag (bool): Indicates whether the current spectrum meets the criteria for saving,
                              based on the measurement configuration and operational logic.

        Note:
            The method internally handles the determination of whether it's the appropriate time to
            save the spectrum (e.g., based on `save_as` parameter and other saving criteria) and
            performs the saving operation accordingly.
        """
        self.log.info("Determining if the current spectrum should be saved.")
        try:
            # Get the latest data
            _ = self.return_data()
            self.log.debug("Latest spectrum data prepared for saving.")

            # Define save methods
            save_methods = {
                "single": self._save_single,
                "kinetic": self._enqueue_kinetic,
                "monitor": self._enqueue_monitor_server,
                "server_single": self._enqueue_save_server,
            }

            # Check if the `save_as` parameter matches a save method
            save_as = self.current_parameters["save_as"]
            if save_as in save_methods:
                self.log.info(f"Saving spectrum using method: {save_as}")
                save_methods[save_as](save_flag)
                self.log.info(f"Spectrum saved successfully using method: {save_as}.")
            else:
                self.log.warning(
                    f"Invalid or unsupported save method: {save_as}. Spectrum not saved."
                )

        except Exception as e:
            self.log.exception(f"Error during spectrum saving process: {e}")
            raise

    def _save_single(self, save_flag):
        """
        Saves the current spectrum as a single file if the conditions are met.

        Parameters:
            save_flag (bool): Indicates whether the current spectrum meets the criteria for saving,
                              based on the measurement configuration and operational logic.

        Note:
            This method is called internally by `save_spectrum_if_needed` to handle the saving of the
            current spectrum as a single file. It checks the conditions for saving and triggers the
            saving operation if the conditions are met.
        """
        self.log.debug(
            "Checking if the current spectrum should be saved as a single file."
        )
        self._enqueue_monitor_server(None)
        self.name_stamp = self.current_time
        self.save_data(save_flag)

    def _enqueue_kinetic(self, _):
        """
        Enqueues the current spectrum for saving in a kinetic measurement mode.

        This method enqueues the current spectrum for saving in a kinetic measurement mode. It adds the
        current spectrum to the data queue for subsequent saving and clears the data frame to only contain
        the 'Rshift' column after enqueuing the spectrum.
        """
        self.log.debug("Enqueuing spectrum for kinetic saving.")
        self.data_queue.put(self.data_df.iloc[:, [0, -1]])

    def _enqueue_monitor_server(self, _):
        """
        Enqueues the current spectrum for monitoring or server saving.

        This method enqueues the current spectrum for monitoring or server saving. It adds the current
        spectrum to the data queue for subsequent processing and clears the data frame to only contain
        the 'Rshift' column after enqueuing the spectrum.
        """
        self.log.debug("Enqueuing spectrum for monitoring or server saving.")
        self.data_queue.put(self.data_df)

    def _enqueue_save_server(self, save_flag):
        """
        enqueues the current spectrum for server saving, saves a copy to the robochem directory.

        """
        self.log.debug("Enqueuing spectrum for server saving.")
        self.data_queue.put(self.data_df)
        self.name_stamp = self.current_time
        self.save_data(save_flag)

    def prepare_final_save(self, save_counter) -> bool:
        """
        Prepares for the final save operation in a kinetic measurement mode as the last scan approaches.

        Parameters:
            save_counter (int): The current count of saved spectra.

        Returns:
            bool: True if the conditions are met for a final save operation in kinetic mode; False otherwise.
        """
        return save_counter == self.current_parameters["n_scans"] - 1

    def save_path_management(self):
        """
        Manages and prepares the save path for data files.

        This method verifies the save path specified in `current_parameters` and constructs the
        filename for saving data. It splits the save path into a directory path and a base filename,
        appending a timestamp to the basename for uniqueness. If no save path is specified, it raises
        an exception.

        Raises:
            Exception: If no save path is specified in `current_parameters`.
        """

        self.log.info("Managing and preparing the save path for data files.")

        try:
            # Check if the save path is specified
            if self.current_parameters["save_path"] is None:
                self.log.error("No save path specified in current parameters.")
                raise Exception("No save path specified, please specify a save path")

            # Set the save path
            self.path = self.current_parameters["save_path"]
            self.log.debug(f"Save path set to: {self.path}")

            # Determine the moniker (base name for the file)
            moniker = self.current_parameters["save_moniker"]
            self.name = moniker if moniker else "spectrum"
            self.log.debug(f"File base name set to: {self.name}")

            # Append timestamp to the filename if not a server-based save mode
            if not self.current_parameters["save_as"] in [
                "server_single",
                "server_kinetic",
            ]:
                self.name = f"{self.name}_{self.current_time}"
                self.log.debug(f"Timestamp appended to file name: {self.name}")

        except Exception as e:
            self.log.exception(f"Error during save path management: {e}")
            raise

    def _reset_data_frame(self):
        """
        Resets the data frame to only contain the 'Rshift' column.

        This method resets the data frame to only contain the 'Rshift' column, effectively removing
        any additional columns that may have been added during the measurement process. It ensures
        that the data frame is clean and ready for subsequent data collection and processing.
        """
        self.log.debug("Resetting the data frame to only contain the 'Rshift' column.")
        self.data_df = self.data_df[["Rshift"]]

    def _enqueue_reset(self):
        """
        Enqueues the current data and resets the data frame
        """
        self.log.debug("Enqueuing the current data and resetting the data frame.")
        self.data_queue.put(self.data_df)
        self._reset_data_frame()

    def _save_to_file(self, mode):
        """
        Saves the current data to a file in the specified format.

        This method saves the current data in `data_df` to a `.csv` file in the directory specified
        by `self.path` and with the filename constructed from `self.name`. After saving, it resets
        the data frame for future measurements.

        Parameters:
            mode (str): The mode of saving, used for additional context in logging or printing.

        Note:
            - The method uses the `self.path` and `self.name` attributes to determine the file path and name.
            - The data frame is reset after saving to prepare for subsequent measurements.

        Raises:
            Exception: If there are any issues during the saving process, such as an invalid path or filename.
        """
        self.log.info(f"Saving data to {self.path} with name {self.name} ({mode}).")
        try:
            # Save the data to a CSV file
            file_path = os.path.join(self.path, f"{self.name}.csv")
            self.data_df.to_csv(file_path)
            self.log.info(f"Data successfully saved to file: {file_path}")

            # Reset the data frame
            self._reset_data_frame()
            self.log.debug("Data frame reset after saving.")

        except Exception as e:
            self.log.exception(f"Error saving data to file: {e}")
            raise

    def _final_save(self):
        """
        Finalizes the saving process by saving the last data (if applicable) and signaling completion.

        This method performs the following:
            - If the `save_as` parameter is set to "server_kinetic", the current data frame (`data_df`)
              is added to the data queue for processing.
            - Signals the completion of the measurement process by adding the keyword "done" to the data queue.
            - Resets the data frame to prepare for the next measurement process.

        Note:
            This method is typically called at the end of a measurement process to ensure all data is saved and
            the system is ready for subsequent tasks.

        Raises:
            Exception: If any issues occur while interacting with the data queue or resetting the data frame.
        """
        self.log.info("Performing final save and signaling completion.")

        try:
            # Save data to the queue if "server_kinetic" mode is enabled
            if self.current_parameters["save_as"] == "server_kinetic":
                self.log.info(
                    "Saving data frame to the queue for 'server_kinetic' mode."
                )
                self.data_queue.put(self.data_df)
                self.log.debug("Data frame added to the queue successfully.")

            # Signal the completion of the measurement process
            self.log.info("Signaling completion by adding 'done' to the data queue.")
            self.data_queue.put("done")

            # Reset the data frame for the next measurement
            self._reset_data_frame()
            self.log.debug("Data frame reset after final save.")

        except Exception as e:
            self.log.exception(f"Error during final save: {e}")
            raise

    @error_handler
    def save_data(self, final_save: bool = False) -> None:
        """
        Saves the spectrum data to a file based on the configured parameters.

        This method handles the logic for saving spectrum data according to the `save_as` parameter.
        It supports different saving modes such as 'single', 'kinetic', and 'monitor'. For 'single' mode,
        each spectrum is saved to a separate file. For 'kinetic' mode, all spectra are saved in one file,
        with the final save operation triggering the actual file write. In 'monitor' mode, data is placed
        in a queue for monitoring purposes and not saved to a file.

        Parameters:
            final_save (bool): Indicates whether this is the final save operation, relevant for 'kinetic' mode.

        Note:
            The method also manages the save path and filename through `save_path_management` and updates the
            data queue as necessary. It resets the data frame to only contain the 'Rshift' column after saving.
        """
        self.log.info("Initiating data save process.")

        try:
            # Manage save path and file name
            self.save_path_management()

            # Handle 'single' and 'server_single' modes (save each spectrum separately)
            if (
                self.current_parameters["save_as"] in ["single", "server_single"]
                and not final_save
            ):
                self.log.info("Saving spectrum in 'single' or 'server_single' mode.")
                self._save_to_file("SINGLE")

            # Handle 'kinetic' and 'server_kinetic' modes (save all spectra in one file)
            elif (
                self.current_parameters["save_as"] in ["kinetic", "server_kinetic"]
                and final_save
            ):
                self.log.info(
                    "Saving spectrum in 'kinetic' or 'server_kinetic' mode (final save)."
                )
                self._save_to_file("KINETIC")

                # Perform final save operations if applicable
                if final_save:
                    self.log.info("Performing final save operations.")
                    self._final_save()

            # Unsupported or unconfigured save mode
            else:
                self.log.warning(
                    f"Save operation skipped. Invalid or unhandled save mode: {self.current_parameters['save_as']}."
                )

        except Exception as e:
            self.log.exception(f"Error during the data save process: {e}")
            raise

    def kill_process(self):
        """
        Terminates the spectrometer process and cleans up resources.

        This method is intended to be called when shutting down the server or application to ensure a clean
        termination. It logs the shutdown process, finalizes the connection with the spectrometer, and performs
        necessary GPIO cleanup operations.
        """
        self.log.info("Killing AVAlanCHE")
        ava.AVS_Done()
        GPIO.cleanup()


if __name__ == "__main__":
    # for debugging purposes on spectrometer:
    spec = AVAlanCHE()
    spec.current_parameters["save_path"] = "/home/pi/Desktop/AVAlanCHE/AVAlanCHE/data"
    spec.current_parameters["save_as"] = "monitor"
    spec.get_dark()
    spec.kill_process()
