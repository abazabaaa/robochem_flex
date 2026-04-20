# 10 — Safety, Error Handling & Recovery

The Robochem Flex platform layers error handling across four tiers: firmware (endstops + error registers), serial-driver (timeouts + ack parsing), unit-task (domain exceptions), and experiment orchestrator (three-tier policy). Observability is via file logs plus a WebSocket stream. This report walks through what actually happens when things go wrong on the robot-control side.

## 1. Error taxonomy

The codebase splits exceptions into two hierarchies: one for physical devices (`devices/errors.py`) and one for procedural unit tasks (`procedures/unit_tasks/errors.py`).

| Exception | Base | Meaning | Raised in |
| --- | --- | --- | --- |
| `DeviceCommunicationError` | `DeviceError` | Serial open/read/write failed, or invalid device response | `device_arduino.py:470, 720, 773`; `open/close/_read/_write` |
| `NoResponseError`, `InvalidResponseError`, `FailedWriteError` | `DeviceCommunicationError` | Specialisations | `devices/errors.py:50–58` |
| `DeviceTimeoutError` | `DeviceError` | Top-level device op did not complete in time | `phase_sensors.py:152, 403, 820` |
| `ParameterTimeoutError` | `ParameterError` | Parameter ack not received in window | `device_arduino.py:667` |
| `ParameterAcknowledgeError` | `ParameterError` | Ack value malformed / unexpected | `device_arduino.py:691` |
| `ParameterCommError` | `ParameterError` | Unexpected serial data while servicing a parameter | `device_arduino.py:627` |
| `SoftLimitError` / `HardLimitError` / `AlarmLockError` | `ParameterError` | grblr (Cartesian sampler) alarms | `devices/nrg/sampler.py:1181, 1189, 1197` |
| `TaskTimeoutError` | `UnitTaskError` | Unit task did not hit its expected signal | `driving/driving_pumps.py:527` |
| `BadSlugQualityError` | `UnitTaskError` | Phase analysis failed: volume off-target | `sensing/phase_sensors.py:586` |
| `CloggingError` | `UnitTaskError` | Too few phase changes — possible blockage | `experiments/chemistry.py:911` |
| `NoSuitableVialError` | `UnitTaskError` | No vial matching type/volume constraints | `sampling/liquid_handler_sampling.py:840–2805` |
| `RecipeError` | `UnitTaskError` | Cannot compose a recipe from available stocks | `sampling/liquid_handler_sampling.py:1412, 1521, 1568` |

## 2. Hardware disconnect — `device_arduino.py`

`ArduinoDevice.open()` sets `_serial_iface.timeout = 1` and `write_timeout = 1` (`device_arduino.py:420–421`). On open it performs a 2 s sleep to ride out the Arduino auto-reset (`:466`). Every `_read`/`_write` operation wraps the underlying `pyserial` call in a try/except that catches `serial.SerialException, ValueError` and re-raises as `DeviceCommunicationError` carrying a reference to the originating device (`:719–724`, `:772–777`). The serial access itself is guarded by `self._serial_thread_lock` (`:710, 765`), so a disconnect never corrupts a concurrent reader.

There is no automatic reconnect inside the driver: `reopen()` exists (`:476–485`) but must be called explicitly. In practice, the reconnect is driven by the experiment's "restart-platform" tier (section below) — the caught `DeviceCommunicationError` bubbles up through the unit task and is caught in `BaseExperiment._run` (`base_experiment.py:837`). `flush_buffer_in()` (`:521`) is called before each transaction to drain stale data and trigger `parse_error()` on unexpected bytes, preventing a phantom response from being mis-parsed as an ack.

## 3. Hard limits — firmware vs Python

Hard limits are enforced *in firmware*, surfaced as *errors* over serial, and *translated* to Python exceptions.

Firmware side: the syringe-pump `AsyncStepperDriver` polls two endstop pins (`AsyncStepper.cpp:202, 234`) and writes `MOTOR_ERROR_ENDSTOP_POS`, `MOTOR_ERROR_ENDSTOP_NEG`, or `MOTOR_ERROR_BLOCKED` (motor stalled against no signal) into `last_error` (`AsyncStepper.h:40–45`). Main loop lifts this into the device-wide `error_type = ERROR_PUMP_MOTOR` register (`syringe_pump_nrg.ino:541–545, 593–596`). The register is polled by the Python side after every write (`device_arduino._verify_error_register`, `:731–749`) and translated to a `DeviceCommunicationError` via the `ArduinoError.exception()` lookup (`:199–229`).

For the grblr-based Cartesian sampler the firmware emits human-readable ALARMs (`"ALARM: Soft limit"`, `"ALARM: Hard limit"`, `"error: Alarm lock"`) which `sampler.py:1180–1204` converts into `SoftLimitError` / `HardLimitError` / `AlarmLockError`. Note: **there is no redundant Python-side range check** — the code trusts firmware enforcement.

## 4. Watchdog & continuous monitoring

`ContinuousMonitoring` (`unit_tasks/sensing/monitoring.py:173–234`) is a task designed to run on its own thread. It holds a list of `MonitoredValue`s, each with a `deque` history and a list of `MonitorEventTrigger` conditions. Every `update_delay` seconds (default 30 s) it reads each value and evaluates the triggers; if a condition fires, its `_on_true: Event` is set, which the main thread can wait on. A stability-check factory `stable_within_range` (`:76–102`) is provided. Exceptions during updates are logged but **do not abort the monitor** (`:233–234` — `cls.log(error, level="warning")`).

Phase sensors provide the other mid-run abort path. `WaitForPhaseChange._execute_wait` (`phase_sensors.py:69–157`) times out after a configurable window and raises `DeviceTimeoutError`; `PumpUntilPhaseChange._execute` raises it if the max volume is reached before phase change (`:403`). `SlugQualityCheck._execute` raises `DeviceTimeoutError` on timeout (`:820`) and `SlugQuality.raise_exception` raises `BadSlugQualityError` when the measured slug volume deviates beyond `max_volume_deviation` (`:586, 746`).

## 5. Emergency stop — Run Platform page

Looking at `Control Software/pages/07_Run_Platform.py:116–134`: the layout allocates four columns including one explicitly named `col_emergency_stop` (`:116`), but **only three buttons are actually created** — Start (`:117`), a commented-out Pause (`:123`), and "Shutdown" (`:129`). There is no wired-up emergency-stop button; the shutdown button is the strongest control surface.

"Shutdown" calls `experiment_shutdown()` (`:92–102`), which calls `backend.stop()`. `platform_backend.stop()` (`platform_backend.py:772–782`) performs three things: clears `self._rolling` (UI state), sets `self._emergency_stop` (an `Event`, `:133`), and calls `self.platform_experiment.stop()`. The last call (`base_experiment.stop`, `:336–353`) sets `_stop_event` and blocks on `_samples_queue`. Crucially, the running run is **allowed to complete** — stop_event is only checked at the top of the run-processing loop (`base_experiment.py:769, 792`). There is no mechanism to interrupt an in-flight device operation; true emergency-off requires the physical power switches mentioned in the on-screen checklist (`07_Run_Platform.py:70`).

## 6. Recipe / slug failures

`BadSlugQualityError`: caught in `chemistry.py:908`. If the failure was due to too few phase changes (`len(slug_shape.volumes) <= 1`) the handler escalates to `CloggingError` (`:911`), which is *not* in the retry list in `base_experiment.py:837` and will therefore halt the platform. Otherwise, the slug is flagged for discard (`self._discard_slug = True`) and `BadSlugQualityError` is re-raised; in the orchestrator it matches the first `isinstance` branch at `base_experiment.py:831` — the run fails but the loop *continues* (swallow-and-retry tier).

`NoSuitableVialError`: when `allow_user_requests` is enabled, the sampling code sends a `UserSetSamplesRequest` through the platform's `user_action_requester` and loops back to retry (`liquid_handler_sampling.py:899–919`). If the requester returns False (user aborted) or is absent, the error propagates. In `base_experiment.py:857` unrecognised exceptions hit the third tier: "The platform will shutdown." is logged and the exception is re-raised, terminating the `_run` thread after `_procedure_shutdown`.

`RecipeError`: raised purely from sampling logic (`liquid_handler_sampling.py:1412, 1521, 1568`) and handled generally as a halt-tier error (no explicit branch in `_run`).

## 7. Observability

The `Logger` (`Control Software/backend/Robochem_ML/robrains/base_classes/logger.py`) is a class-level singleton. On first `log_message` call it lazily spawns a uvicorn subprocess hosting a FastAPI WebSocket server (`:42–75`) and opens a browser window pointing at `http://localhost:6999`. Messages are POSTed to `/log` with a 50 ms timeout (`:264`) — if the server is slow, the post silently drops (`except Exception: pass`, `:267–268`). Messages are also appended to daily `logs/YYYY-MM-DD.txt` files and per-origin subfolders (`:235–272`).

The server (`websocket_logger.py:175–256`) holds a 1000-line ring buffer (`MAX_HISTORY`), an asyncio `LOG_QUEUE`, and a set of connected `CLIENTS`. The `_broadcaster` task fans each queued line out to all websocket clients (`:181–196`). New clients get the full history replayed on connect (`:226–227`). Log levels are tagged `error`, `warning`, `ok` (`:117–121`) and colorized via Colorama. `atexit.register(Logger.shutdown_log_server)` (`:281`) cleans up the subprocess.

## 8. Known gaps / risks

- **No wired emergency-stop button.** `07_Run_Platform.py:116` reserves a column for it but only Shutdown is implemented, which still lets the current run finish — there is no software interrupt that immediately halts motion. Physical cutoff is the only fast abort.
- **Best-effort log transport.** The WebSocket path uses a 50 ms POST timeout and silently swallows failures (`logger.py:264–268`); under load, errors can be dropped from the live stream even though the files still get them. ML-side divergence has no dedicated exception type and relies on generic `ValueError` handling in `base_experiment.py:833`, which silently retries.
- **Restart-loop can mask systemic failure.** The middle tier (`base_experiment.py:835–856`) catches `DeviceCommunicationError`, `HardLimitError`, `AlarmLockError` and unconditionally restarts the platform. There is no retry counter — a repeatedly-failing endstop will loop `build → fail → build` indefinitely, burning reagents and log volume.
- **`ContinuousMonitoring` swallows exceptions.** Errors in `monitored.update()` are logged as warnings only (`monitoring.py:233–234`), so a silently dead sensor cannot itself abort the run; only downstream tasks that depend on the reading will surface the problem.
