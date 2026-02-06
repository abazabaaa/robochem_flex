"""
File: base_experiment.py
Author: Elia Savino, Simone Pilon - Noël Research Group - 2023
GitHub: github.com/EliaSavino, https://github.com/simone16

Description: Base Experiment class. Implements core functions, but no particular experimental procedure.
"""

from abc import ABC, abstractmethod
from typing import Any, Type
import copy
import pandas as pd
from threading import Event, Thread
from queue import Queue, Empty, Full

from omniplatypus.devices.errors import (
    DeviceCommunicationError,
    DeviceTimeoutError,
    ParameterCommError,
    ParameterTimeoutError,
    ParameterAcknowledgeError,
    ParameterNotRunningError,
    SoftLimitError,
    HardLimitError,
    AlarmLockError,
)
from omniplatypus.procedures.unit_tasks.user_actions import (
    UserActionRequest,
    UserAction,
    UserSetSamples,
    UserSetSamplesRequest,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
    VialRecipeComponent,
)
from omniplatypus.procedures.unit_tasks.errors import (
    BadSlugQualityError,
)
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
    ChemicalParameter,
    RunResult,
)
from omniplatypus.procedures.analytics.analytics_template import AnalyticsTemplate
from omniplatypus.devices.platform import Platform
from omniplatypus.utilities.general import conversion_factor, flush_queue
from omniplatypus.utilities.logger import Logger


class ExperimentAnalysisCoupler:
    """
    Holds information to perform a certain analytics procedure in combination with a specific experiment.
    Every experiment should define one of these objects for each supported analytic.
    """

    analysis_class: Type[AnalyticsTemplate] | None
    analytical_device: str | None
    platform_constants_key: str | None

    def __init__(
        self,
        analysis_class: Type[AnalyticsTemplate] | None = None,
        analytical_device: str | None = None,
        platform_constants_key: str | None = None,
    ):
        """
        Constructor.

        @param analysis_class: Type[AnalyticsTemplate]
            Type to use for the analysis.
        @param analytical_device: str | None = None
            Name of analytical device (as found in platform_config.json).
            The analytical device if given to the analysis constructor.
        @param platform_constants_key: str | None = None
            Constants pertaining to the pairing of the analysis to a specific platform (such as volumes and offsets)
            are stored in the platform_config.json file under 'constants' > 'analysis' and this key.
        """
        self.analysis_class = analysis_class
        self.analytical_device = analytical_device
        self.platform_constants_key = platform_constants_key


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments. It outlines the essential structure
    of an experiment, including initialization, running, analysis, and shutdown procedures.

    This class is designed to be subclassed for specific experiment implementations,
    where methods containing the word 'procedure' should be overridden to provide
    specific functionalities.

    Usage:
    Once the experiment is started with `self.start()`, queue runs by calling `self.submit_run()` and
    retrieve results by calling `self.get_results()`.
    Stop the experiment by calling stop(), the running experiment will complete and the platform will be shut down.
    """

    _analytical_methods: dict[
        str, ExperimentAnalysisCoupler
    ] = {}  # Analytical methods supported by the experiment
    _required_parameters: list[
        ExperimentalParameter
    ] = []  # These parameters must be included for each run
    _optional_parameters: list[
        ExperimentalParameter
    ] = []  # These parameters take the default value if unspecified

    _required_devices: set[
        str
    ] = set()  # Names of required devices (see platform_config.json)

    _platform_name: str
    _analytical_method: AnalyticsTemplate | None
    _analysis_coupling: ExperimentAnalysisCoupler | None
    _platform_build_arguments: dict[str, Any] | None
    _platform_constants: dict[str, Any]
    _storage_root: str | None
    _run_requests_queue: Queue
    _run_results_queue: Queue
    _user_action_queue: Queue
    _action_request_queue: Queue
    _samples_queue: Queue
    _stop_event: Event
    _resume_event: Event
    _experiment_thread: Thread | None
    _platform: Platform
    _queue_update_interval: float

    @classmethod
    def get_analytical_methods(cls) -> dict[str, ExperimentAnalysisCoupler]:
        """
        List of supported analytical methods names.

        @return: dict[str, ExperimentAnalysisCoupler]
            Names of the analytical methods which are supported by the experiment.
        """
        return copy.deepcopy(cls._analytical_methods)

    @classmethod
    def get_required_parameters(cls) -> list[ExperimentalParameter]:
        """
        List of parameters which must be set for every run.

        @return: list[ExperimentalParameter]
            List of parameters which must be set for every run.
        """
        return copy.deepcopy(cls._required_parameters)

    @classmethod
    def get_optional_parameters(cls) -> list[ExperimentalParameter]:
        """
        List of parameters which can be set for every run.
        If unset, these parameters default to the values listed here.

        @return: list[ExperimentalParameter]
            List of parameters which can be set for every run.
        """
        return copy.deepcopy(cls._optional_parameters)

    @classmethod
    def get_all_parameters(cls) -> list[ExperimentalParameter]:
        """
        List of all parameters which can be set for every run.
        If unset, some of these parameters default to the values listed here.

        @return: list[ExperimentalParameter]
            List of all parameters which can be set for every run.
        """
        return copy.deepcopy(cls._required_parameters + cls._optional_parameters)

    def __init__(
        self,
        analytical_method: str,
    ) -> None:
        """
        Constructor.

        @param analytical_method: str | None = None
            The analytical method to build the platform with.
            Must match exactly one of the keys of cls._analytical_methods (or None).
        """
        self._platform_name = "<no platform linked>"
        self._analysis_coupling = None
        if analytical_method not in self._analytical_methods.keys():
            error = ValueError(
                f"Requested analytical method '{analytical_method}' "
                f"is not supported by '{self.__class__.__name__}' experiment."
            )
            self._log(error)
            raise error
        self._analysis_coupling = self._analytical_methods[analytical_method]
        self.check_duplicate_parameters()
        self._input_samples = None

        # Thread interface accessible from public methods.
        # To avoid race conditions, public methods should not access any other object member.
        self._run_requests_queue = Queue()
        self._run_results_queue = Queue()
        self._user_action_queue = Queue()
        self._action_request_queue = Queue()
        self._samples_queue = Queue()
        self._stop_event = Event()
        self._resume_event = Event()
        self._experiment_thread = None
        self._queue_update_interval = 5.0

        # Private members, only accessed by experiment thread.
        self._platform = Platform()
        self._platform_build_arguments = None
        self._platform_constants = dict()
        self._analytical_method = (
            None  # this is populated in _procedure_build, as it needs the device.
        )

    def check_duplicate_parameters(self):
        """
        Check whether the current experiment configuration defines different parameters with the same name.

        @raise: ValueError
            If duplicates are found.
        """
        analysis_class = self._analysis_coupling.analysis_class
        all_names = [
            x.name for x in self._required_parameters + self._optional_parameters
        ]
        if analysis_class is not None:
            all_names += [x.name for x in analysis_class.get_all_parameters()]
        unique_names = set(all_names)
        if len(unique_names) < len(all_names):
            analysis_name = ""
            if analysis_class is not None:
                analysis_name = (
                    f" in combination with '{analysis_class.__name__}' analysis"
                )
            error = ValueError(
                f"Experiment '{self.__class__.__name__}'{analysis_name} uses one name for multiple parameters."
            )
            for name in unique_names:
                count = all_names.count(name)
                if count > 1:
                    error.add_note(f"'{name}' is used {count} times.")

    def start(
        self,
        platform_name: str,
        samples: pd.DataFrame,
        platform_build_arguments: dict[str, Any] | None = None,
        override_platform_constants: dict[str, Any] | None = None,
        file_storage_root: str | None = None,
        queue_update_interval: float = 5.0,
    ) -> None:
        """
        Start the thread running the experiments on the platform.

        @param platform_name: str
            The name of the platform running these experiments.
        @param samples: pandas.DataFrame
            Information on the vials and samples available to the platform, as produced by the GUI.
            A copy is made of this dataframe.
        @param platform_build_arguments: dict[str, Any] | None
            Optional arguments to pass to the platform.build() function.
        @param override_platform_constants: dict[str, Any] | None = None
            A dictionary can be passed here. Structure must match that of the 'constants' section of the
            platform_config.json configuration file (not all values must be provided). Provided values will
            override the platform_config.json. Values which do not match the expected structure will be ignored with a
            warning.
        @param file_storage_root: str | None = None
            Path to folder where all data produced by the experiment will be stored.
        @param queue_update_interval: float = 5.0
            Interval in S between attempts to retrieve items from the queue. This corresponds to the maximum delay
            between requests and their execution.
        """
        self._platform_name = platform_name
        self._platform_build_arguments = platform_build_arguments
        if override_platform_constants is not None:
            self._platform_constants = override_platform_constants
        self._storage_root = file_storage_root
        # If another storage path location was no specified, use the same for the platform too.
        if (
            self._platform_build_arguments is None
            or "storage_path" not in self._platform_build_arguments.keys()
            or self._platform_build_arguments["storage_path"] is None
        ) and self._storage_root is not None:
            if self._platform_build_arguments is None:
                self._platform_build_arguments = {}
            self._platform_build_arguments["storage_path"] = self._storage_root
        self._input_samples = copy.deepcopy(samples)
        self._queue_update_interval = queue_update_interval

        if self._experiment_thread is not None and self._experiment_thread.is_alive():
            error = RuntimeError(
                f"{self.__class__.__name__} experiment already running."
            )
            self._log(error)
            raise error
        self._experiment_thread = Thread(
            target=self._run,
            name=f"main experiment thread {self._platform_name}",
            daemon=False,
        )
        self._stop_event.clear()
        self._resume_event.set()
        self._experiment_thread.start()

    def pause(self) -> pd.DataFrame:
        """
        Pause the experiment after the current run is finished.
        The experiment can continue by calling self.resume().
        Note: this function is always blocking, since it returns the samples, which are only available after the
        run is complete.

        @return: pandas.DataFrame
            Returns the current samples dataframe.
            This allows the user to make modifications to the samples while the experiment is paused.
        """
        # Make sure there is nothing else in the samples queue.
        flush_queue(self._samples_queue)
        self._resume_event.clear()
        # Clearing resume_event should cause a new item to be placed in samples_queue.
        # This is the samples configuration after the experiment was paused, which we return.
        samples = self._samples_queue.get()
        self._samples_queue.task_done()
        return samples

    def resume(self, samples: pd.DataFrame | None = None) -> None:
        """
        Resume the experiment after it was paused.

        @param samples: pandas.DataFrame | None = None
            Optional samples dataframe to replace the currently active one.
        """
        self._resume_event.set()

    def stop(self) -> pd.DataFrame:
        """
        Stop the thread running the experiments as soon as the running experiment is complete.

        @return: pandas.DataFrame
            Returns the last state of the samples dataframe.
            This allows the user to save the samples dataframe after the platform is stopped.
        """
        # Make sure there is nothing else in the samples queue.
        flush_queue(self._samples_queue)
        self._resume_event.set()  # this should make the stop a little faster.
        self._stop_event.set()
        # Setting stop should cause a new item to be added to the samples queue, we wait until that is added.
        samples = self._samples_queue.get()
        self._samples_queue.task_done()
        # The thread running the experiment should also stop (after setting stop_event).
        self._experiment_thread.join()
        return samples  # Return the samples dataframe

    def check_run_parameters(self, parameters: list[ExperimentalParameter]) -> None:
        """
        Check validity of parameters list.

        @param parameters: list[ExperimentalParameter]
            Parameters to give to the submit_run function.
        @raise: ValueError
            If the parameters are invalid.
        """
        # exclude chemical parameters from the provided ones.
        physical_parameters_names = {
            parameter.name
            for parameter in parameters
            if not isinstance(parameter, ChemicalParameter)
        }

        # calculate required parameters
        required_parameter_names = {
            parameter.name for parameter in self._required_parameters
        }
        if self._analysis_coupling.analysis_class is not None:
            required_parameter_names = required_parameter_names.union(
                {
                    parameter.name
                    for parameter in self._analysis_coupling.analysis_class.get_required_parameters()
                }
            )

        # raise if missing required parameters
        missing_parameters_names = required_parameter_names.difference(
            physical_parameters_names
        )
        if len(missing_parameters_names) > 0:
            error = ValueError(
                f"Experiment '{self.__class__.__name__}' requires parameters '{missing_parameters_names}',"
                f" but these were not specified."
            )
            self._log(error)
            raise error

        # calculate all parameters
        all_parameter_names = {
            parameter.name
            for parameter in self._required_parameters + self._optional_parameters
        }
        if self._analysis_coupling.analysis_class is not None:
            all_parameter_names = all_parameter_names.union(
                {
                    parameter.name
                    for parameter in self._analysis_coupling.analysis_class.get_all_parameters()
                }
            )

        # raise if too many parameters were provided
        extra_parameters_names = physical_parameters_names.difference(
            all_parameter_names
        )
        if len(extra_parameters_names) > 0:
            error = ValueError(
                f"Experiment '{self.__class__.__name__}' does not recognize parameters '{extra_parameters_names}'."
            )
            self._log(error)
            raise error

        # raise if parameter has incorrect type
        all_experiment_parameters = (
            self._required_parameters + self._optional_parameters
        )
        if self._analysis_coupling.analysis_class is not None:
            all_experiment_parameters += (
                self._analysis_coupling.analysis_class.get_all_parameters()
            )
        all_experiment_parameters = {
            parameter.name: parameter for parameter in all_experiment_parameters
        }
        for parameter in parameters:
            if not isinstance(parameter, ChemicalParameter):
                if type(parameter) is not type(
                    all_experiment_parameters[parameter.name]
                ):
                    error = ValueError(
                        f"Experiment '{self.__class__.__name__}' requires parameter '{parameter.name}' to be of type "
                        f"'{type(all_experiment_parameters[parameter.name])}', but '{type(parameter)}' was provided."
                    )
                    self._log(error)
                    raise error

    def submit_run(
        self, run_id: str, parameters: list[ExperimentalParameter], **kwargs
    ) -> None:
        """
        Add an experiment run to the internal queue.

        @param run_id: str
            The experiment identifier.
        @param parameters: list[ExperimentalParameter]
            The list of physical and chemical conditions to employ in the queued run.
        @param kwargs:
            These are passed to Queue.put() when the experiment is added to the queue.
        @keyword block=True
            If False, return immediately even when the queue is full.
        @keyword timeout=None
            Timeout for waiting for a free spot in the queue.
        @raise: ValueError
            If invalid parameters are provided.
        """
        self.check_run_parameters(parameters)

        # queue run
        run_request = {"ID": run_id, "parameters": copy.deepcopy(parameters)}
        self._run_requests_queue.put(run_request, **kwargs)

    def get_result(self, **kwargs) -> RunResult | None:
        """
        Retrieve the result of the earliest (not yet retrieved) experiment run.
        Note: set the timeout parameter so that enough time passes to complete the experiment. If None is returned
        after an abnormally long time, something went wrong!

        @param kwargs:
            These are passed to Queue.get() when the experiment is added to the queue.
        @keyword block=True
            If False, return immediately even when the queue is empty.
        @keyword timeout=None
            Timeout for waiting for a result to become available in the queue.
        @return: RunResult | None
            The experiment ID and result, or None if the timeout expired and the queue was empty.
        """
        try:
            result = self._run_results_queue.get(**kwargs)
        except Empty:
            return None
        else:
            self._run_results_queue.task_done()
            return copy.deepcopy(result)

    def get_samples(self, **kwargs) -> pd.DataFrame | None:
        """
        Get the oldest samples dataframe submitted.

        @param kwargs:
            These are passed to Queue.get() when the experiment is added to the queue.
        @keyword block=True
            If False, return immediately even when the queue is empty.
        @keyword timeout=None
            Timeout for waiting for a result to become available in the queue.
        @return: pandas.DataFrame | None
            The samples dataframe or None.
        """
        try:
            samples = self._samples_queue.get(**kwargs)
        except Empty:
            return None
        else:
            self._samples_queue.task_done()
            return samples

    def get_action_requests(self, **kwargs) -> UserActionRequest | None:
        """
        Query the internal queue for user action requests and errors.
        A user action typically blocks the experiment from running until the error condition is resolved by the user.
        Once an action request is obtained, the user must perform some action to fix the condition and communicate the
        fix to the experiment (via self.. For now all fixes are sent using the update_samples() function to alter the samples
        dataframe.

        @param kwargs:
            These are passed to Queue.get() when the experiment is added to the queue.
        @keyword block=True
            If False, return immediately even when the queue is empty.
        @keyword timeout=None
            Timeout for waiting for a result to become available in the queue.
        @return: UserActionRequest | None
            A UserActionRequest object with information on the error condition and what to do to resolve it, or None if
            all is good.
        """
        try:
            action_request = self._user_action_queue.get(**kwargs)
        except Empty:
            return None
        else:
            self._user_action_queue.task_done()
            return action_request

    def submit_user_action(self, resolution: UserAction, **kwargs) -> None:
        """
        Queues a request to update the status of the machine after an external change.

        @param resolution: UserAction
            Object with the required data to update the status.
        @param kwargs:
            These are passed to Queue.put() when the error resolution is added to the queue.
        @keyword block=True
            If False, return immediately even when the queue is full.
        @keyword timeout=None
            Timeout for waiting for a free spot in the queue.
        """
        self._action_request_queue.put(resolution, **kwargs)

    def _process_user_actions(self) -> bool:
        """
        Empty the user action queue by processing each UserAction within it.

        @return: bool
            True if at least one UserAction was processed.
        """
        nothing_was_processed = True
        while True:
            try:
                action: UserAction = self._action_request_queue.get(block=False)
            except Empty:
                break
            action_type = type(action)
            if action_type is UserSetSamples:
                action: UserSetSamples
                try:
                    self._procedure_update_samples(action.samples)
                except ValueError as error:
                    # re-send the request, as there was a problem with processing it.
                    self._user_action_queue.put(
                        UserSetSamplesRequest(
                            current_samples=action.samples,
                            description="An error occurred while processing your fix for the last issue:\n"
                            + str(error),
                        )
                    )
                else:
                    nothing_was_processed = False
            else:
                self._log(
                    f"Cannot handle '{action_type.__name__}' as a UserAction.",
                    level="error",
                )
            self._action_request_queue.task_done()
        return not nothing_was_processed

    def _wait_for_user_action(self, request: UserActionRequest) -> bool:
        """
        Submit a UserActionRequest and wait until a UserAction is processed which solves the problem.
        Note: This is meant to be called by platform or unit tasks in the _run thread.

        @return bool:
            True if the caller should try again, False if there is no hope (experiment was stopped).
        """
        # send the request
        self._user_action_queue.put(request)

        # wait for a status update while the experiment is not stopped.
        while not self._stop_event.is_set():
            self._stop_event.wait(self._queue_update_interval)
            if self._process_user_actions():
                return True
        return False

    def _merge_dicts(self, override: dict, overridden: dict):
        """
        Merge two dictionaries by overriding the values in the second with those in the first.
        Search recursively through nested dictionaries.
        """
        for key, value in override.items():
            if key in overridden.keys():
                if isinstance(value, dict) and isinstance(overridden[key], dict):
                    self._merge_dicts(value, overridden[key])
                elif isinstance(value, dict) or isinstance(overridden[key], dict):
                    self._log(
                        f"Dict structure mismatch for '{key}': '{value}' and '{overridden[key]}'",
                        level="warning",
                    )
                else:
                    overridden[key] = value
            else:
                self._log(
                    f"Attempt to override constant '{key}' failed: does not match a value in platform constants.",
                    level="warning",
                )

    def _merge_platform_constants(self, override: dict) -> None:
        """
        Loads the constants from the built platform onto self._platform_constants while
        keeping any values which are meant to be overridden.
        """
        original_constants = self._platform.constants
        self._merge_dicts(self._platform_constants, original_constants)
        self._platform_constants = original_constants

    def _procedure_build(self) -> None:
        """
        This is called once before the first experiment.
        It must prepare the platform and experiment for operation.
        """
        args = self._platform_build_arguments
        # override name, must match the one explicitly given.
        args["platform_name"] = self._platform_name
        # ensure we have all required devices, in addition to what the user specified.
        devices = copy.deepcopy(self._required_devices)
        if "devices" in args.keys():
            devices.union(set(args["devices"]))
        # add analysis device
        analysis_device_name = self._analysis_coupling.analytical_device
        if analysis_device_name is not None:
            devices.add(self._analysis_coupling.analytical_device)
        args["devices"] = devices
        # Build platform
        self._platform.build(**args)
        self._merge_platform_constants(self._platform_constants)
        # Create Analysis
        analysis_device = None
        if analysis_device_name is not None:
            analysis_device = self._platform[analysis_device_name]
        if self._analysis_coupling.analysis_class is not None:
            self._analytical_method = self._analysis_coupling.analysis_class(
                analytical_device=analysis_device,
                processing_method=None,  # todo get it from GUI!
                storage_root=self._storage_root,
            )

    @abstractmethod
    def _procedure_update_samples(self, samples: pd.DataFrame) -> None:
        """
        Should update the platform dataframe based on a new version of the dataframe provided by the user.

        @param samples: pandas.DataFrame
            An updated version of the samples dataframe.
        """
        pass

    @abstractmethod
    def _procedure_prepare(self) -> None:
        """
        Thorough clean up and prep procedure for the platform. Ensures it's ready to start a new run after a period
        of inactivity.
        This is called before the first run.
        This method must be implemented by non-dummy experiments.
        """
        pass

    @abstractmethod
    def _procedure_experiment(
        self,
        results: RunResult,
        conditions: dict[str, ExperimentalParameter],
        recipe: list[RecipeComponent],
    ) -> None:
        """
        This method runs the experimental procedure, it must be implemented in the subclasses.
        Note that cleanup procedure should be implemented separately (see self._procedure_cleanup).
        This method must be implemented by non-dummy experiments.

        @param results: RunResult
            Any information acquired during the experiment and which needs to be accessed later, such as yield, spectra
            or other, should be stored here. Once the experiment is complete this object is returned.
        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run are given here.
        @param recipe: list[ChemicalParameter]
            Chemical conditions, reagents and their concentrations, are given here.
        """
        pass

    @abstractmethod
    def _procedure_cleanup(self) -> None:
        """
        Clean up procedure for the platform. Ensures it's ready to start a new run.
        This is called after every run.
        This method must be implemented by non-dummy experiments.
        """
        pass

    def _procedure_shutdown(self) -> None:
        """
        Shut down procedure of the experimental platform, ensuring all devices are properly
        turned off or reset.
        """
        self._platform.clear()

    def _run_procedure_experiment(
        self, run_id: str, parameters: list[ExperimentalParameter | NumericalParameter]
    ) -> RunResult:
        """
        Run _procedure_experiment() within the correct environment.
        Calls _procedure_experiment() and handles creation of run results and other common aspects to all procedures.
        Ensures exceptions are caught and a result is always returned.

        @param run_id: str
            Identifier for the run.
        @param parameters: list[ExperimentalParameter]
            List of parameters for the run.
        @return: RunResult
            The result values for the run.
        """
        # set up empty result
        run_result = RunResult(run_id)
        # Physical conditions (non-chemical parameters)
        conditions = self._conditions_from_parameters(parameters)
        # Chemical recipe (chemical parameters)
        recipe = self._recipe_from_parameters(parameters)
        try:
            self._procedure_experiment(run_result, conditions, recipe)
        except Exception as error:
            # Any error encountered is stored in the result
            run_result.exception = error
            run_result.success = False  # this just for good measure
        finally:
            # Important: since the function is returning here, the exception is not raised!
            # to raise in case of failure, check run_result.exception (is None if all was good).
            self._update_parameters(parameters, conditions, recipe)
            run_result.parameters = parameters
            return run_result

    def _run(self):
        """
        Main experiment thread.
        Listens and checks the queues and the events. When a new experiment run is submitted to the
        input queue it will run it and put the results on the output queue. If the stop event is set it will stop at the
         end of the run.
        """
        restart_platform = True
        while not self._stop_event.is_set() and restart_platform:
            restart_platform = False
            try:
                # One-off procedures
                self._log("Building platform...")
                self._procedure_build()
                self._log("Built platform.", level="ok")
                self._log("Preparing platform...")
                self._procedure_update_samples(self._input_samples)
                self._procedure_prepare()
                self._log("Prepared platform.", level="ok")

                # show defaults, as these will not be shown for each run unless actively modified:
                # (defaults can change over time, this way we keep track).
                default_parameters = "Defaults:"
                for parameter in self._optional_parameters:
                    default_parameters += f"\n{parameter}"
                if self._analytical_method is not None:
                    for parameter in self._analytical_method.get_optional_parameters():
                        default_parameters += f"\n{parameter}"
                self._log(default_parameters)

                # Process runs as long as stop is clear.
                while not self._stop_event.is_set():
                    # even if paused, keep processing user actions.
                    self._process_user_actions()
                    if not self._resume_event.is_set():
                        self._resume_event.wait(self._queue_update_interval)
                        # when resumed, start over.
                        continue
                    # wait until a run is available
                    try:
                        run_request = self._run_requests_queue.get(
                            block=True, timeout=self._queue_update_interval
                        )
                    except Empty:
                        # if no run is available, start over (checks stop and resume events).
                        continue
                    else:
                        # Process run...
                        self._log_run_parameters(
                            run_id=run_request["ID"],
                            parameters=run_request["parameters"],
                        )
                        # note: this will typically not raise, but store exception in result.exception
                        result = self._run_procedure_experiment(
                            run_id=run_request["ID"],
                            parameters=run_request["parameters"],
                        )
                        self._run_results_queue.put(result)
                        try:
                            self._samples_queue.put(
                                copy.deepcopy(self._platform.samples)
                            )
                        except Full:
                            self._log(
                                "Platform samples queue is full!", level="warning"
                            )
                        self._run_requests_queue.task_done()
                        if result.exception is not None:
                            # decide what to do if run failed
                            # if the run should be retried or discarded, ignore the exception.
                            if isinstance(result.exception, BadSlugQualityError):
                                pass
                            elif isinstance(result.exception, ValueError):
                                pass
                            elif isinstance(
                                result.exception,
                                (
                                    DeviceCommunicationError,
                                    DeviceTimeoutError,
                                    ParameterCommError,
                                    ParameterTimeoutError,
                                    ParameterAcknowledgeError,
                                    ParameterNotRunningError,
                                    SoftLimitError,
                                    HardLimitError,
                                    AlarmLockError,
                                ),
                            ):
                                # these device errors cause the platform to restart.
                                self._log("An exception was caught during a run:")
                                self._log(result.exception)
                                self._log(
                                    "The platform will restart and continue operations."
                                )
                                restart_platform = True
                                break
                            else:
                                # if the error is not recognized, we halt everything.
                                self._log("An exception was caught during a run:")
                                self._log(result.exception)
                                self._log("The platform will shutdown.")
                                raise result.exception
                        self._log(
                            f"Completed run {result.run_id}, results:\n{result}",
                            level="ok",
                        )
                        self._log("Cleanup...")
                        self._procedure_cleanup()
                        self._log("Cleanup complete.", level="ok")
            finally:
                self._log("Shutting down platform...")
                self._procedure_shutdown()
                self._samples_queue.put(copy.deepcopy(self._platform.samples))
                self._log("Shut down platform.", level="ok")

    def _conditions_from_parameters(
        self, parameters: list[ExperimentalParameter]
    ) -> dict[str, ExperimentalParameter | NumericalParameter]:
        """
        Handy function to convert the list of parameters provided for the run into a dictionary for fast lookup.
        Adds the defaults values for all parameters which are not explicitly added by user.

        @param parameters: list[ExperimentalParameter]
            The list of physical and chemical conditions to employ in the queued run.
        @return: dict[str, ExperimentalParameter]
            Dictionary of experimental conditions excluding chemicals.
        """
        conditions = {
            parameter.name: parameter
            for parameter in parameters
            if not isinstance(parameter, ChemicalParameter)
        }
        missing = {parm.name for parm in self._optional_parameters}.difference(
            set(conditions.keys())
        )
        if len(missing) > 0:
            defaults = {x.name: x for x in self._optional_parameters}
            for name in missing:
                conditions[name] = copy.deepcopy(defaults[name])
        return conditions

    def _recipe_from_parameters(
        self, parameters: list[ExperimentalParameter]
    ) -> list[RecipeComponent]:
        """
        Handy function to convert the list of parameters provided for the run into a recipe to create a slug.
        The recipe returned can be provided as-is to the sampling unit tasks (GenerateComposition).
        Note: the experimental parameters must specify exactly 1 chemical by its concentration, the rest is calculated
        from its equivalents. Alternatively, all chemicals can be given in terms of concentration.

        @param parameters: list[ExperimentalParameter]
            The list of physical and chemical conditions to employ in the queued run.
        @return: list[RecipeComponent]
            List of concentrations for each chemical compound requested for the slug.
        """
        chemicals = [
            parameter
            for parameter in parameters
            if isinstance(parameter, ChemicalParameter)
        ]

        # In this context the limiting reagent is the one whose amount is given as a concentration.
        # The amounts of the other reagents are calculated according to the equivalents of this one.
        # If all reagents are given in concentration, then that is ok. If more concentrations are available but not all
        # then it is a problem.

        limiting_reagents = [
            chemical for chemical in chemicals if chemical.as_concentration()
        ]
        if len(limiting_reagents) == 1:
            # do they all have same priority?
            priorities = set([chemical.sampling_priority for chemical in chemicals])

            limiting = limiting_reagents[0]
            limiting_concentration = limiting.with_units(
                RecipeComponent.concentration_units
            )

            recipe = [
                RecipeComponent(
                    name=limiting.name,
                    concentration=limiting_concentration,
                    sampling_priority=limiting.sampling_priority,
                )
            ]
            # if all chemicals have same priority, make sure the limiting is taken first.
            if len(priorities) <= 1:
                recipe[0].sampling_priority += 100
            for chemical in chemicals:
                if chemical is not limiting:
                    recipe.append(
                        RecipeComponent(
                            name=chemical.name,
                            concentration=chemical.with_units("eq")
                            * limiting_concentration,
                            sampling_priority=chemical.sampling_priority,
                        )
                    )
            return recipe
        elif len(limiting_reagents) == len(chemicals):
            recipe = []
            for chemical in chemicals:
                recipe.append(
                    RecipeComponent(
                        name=chemical.name,
                        concentration=chemical.with_units(
                            RecipeComponent.concentration_units
                        ),
                        sampling_priority=chemical.sampling_priority,
                    )
                )
            return recipe
        else:
            error = ValueError(
                f"Multiple chemicals were provided as limiting (or none, but not all): '{limiting_reagents}'."
            )
            self._log(error)
            raise error

    # noinspection PyMethodMayBeStatic
    def _update_parameters(
        self,
        parameters: list[ExperimentalParameter],
        conditions: dict[str, ExperimentalParameter | NumericalParameter],
        recipe: list[RecipeComponent],
    ) -> None:
        """
        Update the run input parameter list to reflect the values in the conditions and recipe.
        In case these values were altered during the experiment, the run result will indicate so in the parameter list.

        @param parameters: list[ExperimentalParameter]
            Run input parameter list. The values will be modified.
        @param conditions: dict[str, ExperimentalParameter | NumericalParameter]
            Experimental conditions.
        @param recipe: list[RecipeComponent]
            Chemical conditions.
        """
        # Find back limiting reagent, if provided
        chemicals = [
            parameter
            for parameter in parameters
            if isinstance(parameter, ChemicalParameter)
        ]
        limiting_reagents = [
            chemical for chemical in chemicals if chemical.as_concentration()
        ]
        limiting_reagent = None
        if len(limiting_reagents) == 1:
            limiting_reagent = limiting_reagents[0]

        # create a dictionary from the recipe for quick access
        recipe_dict = {component.name: component for component in recipe}

        # Update each parameter based on type.
        for parameter in parameters:
            if isinstance(parameter, ChemicalParameter):
                # For chemicals, we need to convert concentrations back in relative amounts, to match the input units.
                value = recipe_dict[parameter.name].concentration
                units = RecipeComponent.concentration_units
                if not parameter.as_concentration():
                    # Calculate relative amount as equivalents.
                    if limiting_reagent is None:
                        raise ValueError(
                            f"Failed to convert '{parameter.name}' value back to relative amount:"
                            " Cannot find limiting reagent!"
                        )
                    value = value / limiting_reagent.with_units(
                        RecipeComponent.concentration_units
                    )
                    units = "eq"
                parameter.value = value * conversion_factor(units, parameter.units)
            elif isinstance(parameter, NumericalParameter):
                # For numerical parameters, simply convert the value to input units.
                parameter.value = conditions[parameter.name].with_units(parameter.units)
            else:
                # For generic parameters, update value only.
                parameter.value = conditions[parameter.name].value

    def _log(self, message: str | Exception, **kwargs) -> None:
        """
        Log a message.
        Note: relies on platform logger, if the platform is not built, the messages will not be processed.

        @param message: str | Exception
            The message to log, or an exception.
        """
        kwargs["subfolder"] = "experiments"
        if "origin" in kwargs.keys():
            kwargs.pop("origin")
        kwargs["priority"] = 6  # corresponds to low verbosity
        Logger.log_message(message, origin=self.__class__.__name__, **kwargs)

    def _log_run_parameters(
        self, run_id: str, parameters: list[ExperimentalParameter]
    ) -> None:
        """
        Log the parameters of a run

        @param run_id: str
            Identifier for the run.
        @param parameters: list[ExperimentalParameter]
            The parameters to use for the run.
        """
        # self._log(f"Starting run '{run_id}':")
        # for parameter in parameters:
        #     self._log(f"    {parameter}", decorate=False)
        message = f"Starting run '{run_id}':"
        for parameter in parameters:
            message += f"\n{parameter}"
        self._log(message)
