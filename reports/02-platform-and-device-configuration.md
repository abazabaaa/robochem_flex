# 02 — Platform Configuration & Device Instantiation

This report traces how robochem_flex is told *which* physical robot it is driving: how platform specs are discovered on disk, how the `Platform` class parses them, how each device is located on the USB/network bus, and how the Streamlit UI feeds a platform choice down into this machinery. The short version: a single `platform_config.json` file describes each named robot (e.g. `Perry`) as a tree of devices, each device maps to a Python class via `device_constructors`, each device is connected either by a *known-device* serial lookup or an `IP`/`host`/`port`, and `PlatformBackend` is the Streamlit-side orchestrator that selects one of these platforms at runtime.

## 1. Platform discovery

Platform specs live in a single JSON file under the `OmniPlatypus` package:

- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/config/platform_config.json` (the platform catalogue)
- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/config/known_devices.json` (USB/serial ID registry)
- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/config/sample_holder_types.json` (vial-holder geometries)
- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/config/platform_info.json`

The canonical path is resolved by `path_to_configuration_folder()` in `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/utilities/general.py:24-26`:

```python
def path_to_configuration_folder():
    return os.path.join(os.path.dirname(getfile(this_module)), "config")
```

`Platform.default_config_path()` (`devices/platform.py:161-169`) joins that folder with `platform_config.json`. `PlatformBackend` does not use this resolver; instead it hard-codes a relative path (`platform_backend.py:98-100`):

```python
platform_config_path = os.path.join(
    "omniplatypus", "omniplatypus", "omniplatypus", "config", "platform_config.json"
)
```

The config file is a flat dict of `platform_name -> platform_data`. `PlatformBackend.available_platforms` (`platform_backend.py:175-181`) simply returns `list(self.platform_config.keys())`. Each platform block may expose a `Gui_Specs.Supported_Experiments` list that `platform_available_experiments()` (`platform_backend.py:183-212`) intersects with the class-level `platform_constructors` dict to give the UI the experiment menu for the selected robot.

## 2. The `Platform` class lifecycle

The `Platform` class (`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/platform.py:57-717`) owns the live device map. Its public contract:

- `devices: dict[str, BaseDevice | ProxyDevice]` — every device addressable by string name.
- `sample_holders: dict` — metadata for physical vial racks.
- `samples: pd.DataFrame | None` — the per-vial chemistry content.
- `constants` (`platform.py:149-159`) — a deep copy of the `constants` block from the platform section, accessible to experiments.

The central registry is the class attribute `device_constructors` (`platform.py:101-115`), which maps the `"class"` string in the JSON to a concrete Python class:

```python
device_constructors = {
    "back_pressure_regulator": BackPressureRegulator,
    "syringe_pump_nrg": SyringePump,
    "liquid_handler_sampler": Sampler,
    "mass_flow_controller": MassFlowController,
    "solar_light_source": SolarLightSource,
    "e5_c_pid_control": TemperatureControl,
    "phase_sensor_array": PhaseSensorArray,
    "gpio_array": GpioArray,
    "light_array": LightArray,
    "heating_plate_ika": HeatingPlate,
    "rama_berry": RamaBerry,
    "spinsolve_client": SpinsolveClient,
    "hplcclient": HPLCClient,
}
```

`Platform.build()` (`platform.py:348-450`) is the entry point. Given `platform_name`, it:

1. Starts the logging thread (optionally with Slack, a GUI, and a log path).
2. Calls `get_platform_config()` to pull the platform block from the JSON (`platform.py:248-287`).
3. Reads the `devices` dict and snapshots `constants`.
4. Loads `sample_holder_types.json` and calls `parse_holder_types()` (`platform.py:195-214`) to fill `sample_holders`.
5. For each device name, spawns a `Thread` running `build_device()` and awaits them via `run_all()` (defined in `utilities/general.py:235-257`, which raises the first thread-level exception in the caller).
6. Optionally regenerates per-platform docs into `doc/platforms/<name>/`.

`build_device()` (`platform.py:538-642`) is the per-device pipeline and has four steps: *construct → connect → setup → initialize*:

```python
constructor_name = device_data["class"].lower()
device_controller: BaseDevice = Platform.device_constructors[constructor_name]()
device_controller.tags = copy.deepcopy(device_data["tags"])
device_controller.specific_name = device_name
self.devices[device_name] = device_controller
if "serial" in device_data.keys():
    self._connect_serial_device(device_data["serial"], device_controller)
elif "IP" in device_data.keys():
    self._connect_IP_device(device_data["IP"], device_controller)
device_controller.setup(**device_data.get("setup", {}))
device_controller.initialize()
```

Device access uses `Platform.__getitem__` (`platform.py:644-655`), so experiments write `platform["Main_Pump_1"]`. The reverse lookup is `device_id()` (`platform.py:657-670`). `clear()` (`platform.py:689-713`) closes every open device in parallel threads and stops the logging thread; `__del__` calls `clear()` on garbage collection.

Error propagation: every failure inside a builder thread is attached with an `add_note(...)` trail and re-raised in the parent thread (`run_all`), so a bad COM port or wrong `known_device` aborts the whole build rather than silently leaving a half-connected platform.

## 3. Configuration-file format

A platform entry has three named sub-blocks plus (optional) `Gui_Specs`:

- `constants` — platform-wide numeric constants used by experiments (volumes, flow rates, phase-sensor name to use for a given analytical technique). Example keys: `analysis.NMR.reactor_to_analysis_volume`, `cleaning.gas_purge_flowrate`.
- `devices` — mandatory. Each device entry requires `class` (`platform.py:559-577` enforces this) and either `serial` or `IP` (`platform.py:591-601` enforces this, otherwise raises `DeviceError("No connection method specified")`).
- `Gui_Specs` — read only by the Streamlit backend; Omniplatypus itself ignores it.

Per-device mandatory/common fields (documented in the `Platform` docstring at `platform.py:75-86`):

- `tags`: list[str] — user-meaningful categories used for UI filtering (e.g. `"handler"`, `"liquid"`, `"sensor"`, `"analytics"`).
- `class`: str — key into `device_constructors` (case-insensitive; `.lower()` is applied).
- `serial.known_device` OR `IP.host` + `IP.port` — exactly one of the two connection blocks.
- `setup`: dict — forwarded verbatim as kwargs to `device.setup(**setup)`. Fields are device-specific (see each markdown in `Control Software/doc/platforms/Perry/`).

For `ArrayDevice` subclasses (`LightArray`, `PhaseSensorArray`, `GpioArray`), the `setup.array` dict names each sub-channel; after `setup()` returns, `build_device()` walks `device_controller.proxies()` (`platform.py:614-626`) and registers a `ProxyDevice` for each sub-unit in `self.devices` under its own name. This is why `"ps_nmr_in"` is a top-level key in `platform.devices` even though it is physically a channel on `Phase_Sensor_Array_2`.

## 4. Address/port assignment

Serial assignment is indirection-based: the JSON says `"known_device": "main_pump_op1"`, and the real COM port is discovered at runtime by `KnownDevices` (`devices/serial_id.py:19-240`). On `Platform._connect_serial_device` (`platform.py:452-499`):

```python
comport = self._known_devices[known_device_name]
if comport is not None:
    device_controller.open(comport=comport)
```

`KnownDevices.__getitem__` (`serial_id.py:216-240`) enumerates `serial.tools.list_ports.comports()` and picks the first connected device matching the stored `serial_number` / `vid` / `pid` / `manufacturer` / optional `custom_id`. Example entry from `known_devices.json`:

```json
"main_pump_op1": {
    "serial_number": "44236313735351702041",
    "vid": 9025,
    "pid": 67,
    "manufacturer": "Arduino LLC (www.arduino.cc)"
}
```

For otherwise-identical Arduino boards without a unique serial number, `custom_id` (`serial_id.py:100-145`) is read over the open serial port by sending `"?"`. An interactive helper, `KnownDevices.interactive_serial_scan()` (`serial_id.py:253-323`), lets an operator register new Arduinos from the command line and persist them to `known_devices.json`.

IP/network devices use `_connect_IP_device()` (`platform.py:501-536`), which requires `host` and `port` and passes them to `device.open(host=..., port=...)`. Raman (`RamaBerry`), NMR (`SpinsolveClient`) and HPLC (`HPLCClient`) use this path; the Raman block even carries `ssh_host` and `keys` in addition.

## 5. `SampleHolder` and vial positions

`SampleHolder` (`platform.py:43-54`) is a minimal dataclass-like object holding `min_volume`, `max_volume`, `min_depth`, `max_depth`, and `components` (dict of label → offset). `parse_holder_types()` fills `platform.sample_holders` from `sample_holder_types.json`. Each entry defines a 2D grid of sub-positions:

```json
"vial_GC_4ml_4x4": {
  "components": {
    "A": [-33.75, 0], "B": [-11.25, 0], ... ,
    "1": [0, -33.75], "2": [0, -11.25], ...
  },
  "min_volume": 1000.0, "max_volume": 4000.0,
  "min_depth": 11.0, "max_depth": 44.0
}
```

The `Sampler` device (class `liquid_handler_sampler`) binds logical holders to physical coordinates via the `setup.locations` block in the platform config. For example in the Perry config, `Sampler_cnc.setup.locations.holder_A` gives the `(x, y)` for the corner of holder A and a `shape: "vial_GC_4ml_4x4"` which names the grid defined in `sample_holder_types.json`. `PlatformBackend.get_handlers()` (`platform_backend.py:370-410`) crosses the two: it finds every device with a `"handler"` tag, then for each location enumerates the alphanumeric `A1`, `A2`, ... grid positions (`sample_holder_positions()` uses `itertools.product` over alphabetic × numeric component keys, `platform_backend.py:352-368`). The UI uses this to offer the user a valid vial position pull-down.

## 6. `PlatformBackend` vs `Platform` — responsibilities

`Platform` is physical. It opens COM ports, talks to Arduinos, runs HPLC client sockets, and guards the device lifecycle. It has no awareness of Streamlit, ML runs, or session files.

`PlatformBackend` (`Control Software/backend/platform_backend.py:58-826`) is the UI-side orchestrator. It is created once per Streamlit session (`robochem_flex.py:23-24`):

```python
if "platform_backend" not in st.session_state:
    st.session_state["platform_backend"] = PlatformBackend(st.session_state)
```

Its job is to: (a) own the raw `platform_config` dict (loaded in `__init__`, `platform_backend.py:162-163`) so UI pages can query `available_platforms`, supported experiments and handler inventories without instantiating hardware; (b) hold the `SessionContainer` that persists user choices and vial/stock DataFrames; (c) own the `ml_experiment_class` and the `platform_experiment` (a `BaseExperiment`); and (d) drive `start`/`pause`/`stop`. The actual hardware `Platform` is only built inside `initialise_platform()` (`platform_backend.py:693-744`) by `self.platform_experiment.start(platform_name=...)`, which in turn calls `Platform.build()` under the hood with `open_gui`, `allow_slack_messages`, `override_platform_constants` and `file_storage_root`. On MacOS, `gui_open` is forced off because the GUI must run on the main thread (`platform_backend.py:716-720`).

## 7. Streamlit wiring

`robochem_flex.py` is the Streamlit home page; its only job related to platforms is constructing the backend and seeding `st.session_state`. The actual platform choice happens on `pages/02_User_Settings.py:86-98`:

```python
platform_name = col31.selectbox(
    "Select the platform you are using",
    backend.available_platforms,
    key="platform_name",
    ...
)
backend.session_container.update_session("platform_name", platform_name)
```

The following `selectbox` calls `backend.platform_available_experiments(platform_name)` to populate the experiment dropdown (lines 100–117). Those two strings — `platform_name` and `platform_experiment` — are the keys that eventually drive `PlatformBackend.initialise_platform()` → `BaseExperiment.start(platform_name=...)` → `Platform.build(platform_name, ...)`. They are also what appear at the top of every saved session file, e.g. `Examples/CS1/session.json:4-5`:

```json
"platform_name": "Perry",
"platform_experiment": "PhotochemicalReaction",
```

Loading a session file via the uploader in `robochem_flex.py:49-61` therefore restores the full platform + experiment selection before the user has even opened page 02, which is what makes the example campaigns in `Examples/CS*/` reproducible on any machine whose `known_devices.json` matches the Perry hardware.
