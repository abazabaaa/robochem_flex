# 03 — Unit Task → Wire Trace

This report follows two unit-task objects from the orchestration layer down to the ASCII bytes on the serial wire and the matching firmware handler. The protocol is a line-oriented ASCII dialogue at 9600 baud, defined by `ArduinoDevice` in `device_arduino.py`: writes take the form `Sx=y\n`, reads take the form `Rx\n`, where `x` is a per-parameter integer id.

## Example 1 — `PumpVolume` → syringe-pump `pump` (variable 9)

### a. Unit task

`PumpVolume._execute_pump` (the template method invoked by `run`) computes what to pump and writes it to the device using `dict`-style assignment. The interesting write is `pump["pump"] = volume` (uL).

`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/procedures/unit_tasks/driving/driving_pumps.py:609-718`:

```python
pump["flowrate"] = flowrate
if pump.has_valve():
    pump["valve_setpoint"] = valve_position
pumping_event.set()
pump["pump"] = volume            # <-- the pump command
pumping_event.clear()
actual_volume_pumped = available_volume - pump["volume"]
```

### b. Device method

There is no specialized "pump()" method on `SyringePump`; the `__setitem__` dispatch lands in the subclass `_write`, which applies pre-write safety checks (available volume, syringe capacity), adjusts the acknowledge timeout to the expected pumping duration, then delegates to `ArduinoDevice._write`.

`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/nrg/syringe_pump.py:855-896`:

```python
def _write(self, parameter, value):
    if parameter is self._parameter_pump:
        current_volume = self.__getitem__("volume")
        if current_volume - value < 0.0: ...   # raise ParameterError
    if parameter.write_acknowledge:
        if parameter is self._parameter_pump:
            parameter.acknowledge_timeout = (
                self.pumping_time(value) * self._timeout_multiplier)
    ArduinoDevice._write(self, parameter, value)
```

### c. `BaseDevice.__setitem__`

`BaseDevice.__setitem__` performs access/enabled/type checks, calls `parameter.validate(value)`, then `self._write(parameter, value)` (device-specific).

- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/base/device.py:897-936` — `__setitem__` implementation (checks + `_write` call at line 934).
- `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/base/device.py:790-850` — `_check_device_parameter` that gates the call.

The `pump` parameter is `ArduinoParameter(name="pump", internal_id=9, write_acknowledge=True, acknowledge_value="k", stop_command="S21=RUN\n")` — registered in `syringe_pump.py:359-372`.

### d. Serial write

`ArduinoDevice._write` takes the lock, formats the command, and writes ASCII bytes. The command string is built by `_command_write`:

`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/base/device_arduino.py:549-566`:

```python
def _command_write(self, parameter, value):
    if parameter.to_serial_string is not None:
        value_string = parameter.to_serial_string(value)    # float -> "250.0"
    elif isinstance(value, ArduinoValue):
        value_string = value.to_serial_string()
    else:
        value_string = str(value)
    return f"S{parameter.internal_id}=" + value_string + "\n"
```

The actual `write` call is at `device_arduino.py:768-771`:

```python
command = self._command_write(parameter, value)
self._serial_iface.write(command.encode("ascii"))
if parameter.write_acknowledge:
    self._receive_acknowledge(parameter)
```

For `PumpVolume.run(pump, volume=250.0)` the 7 bytes pushed onto the wire are: `S9=250.0\n` (that is `0x53 0x39 0x3D 0x32 0x35 0x30 0x2E 0x30 0x0A`). `pyserial` is configured at 9600 baud, 1 s write/read timeout in `ArduinoDevice.__init__` (`device_arduino.py:418-421`).

### e. Firmware handler

The Arduino UNO parses `S9=...` in the write switch statement. Variable 9 is documented at the top of the file (`syringe_pump_nrg.ino:23`) and handled at:

`Devices/Syringe Pump/Firmware/syringe_pump_nrg/syringe_pump_nrg.ino:424-436`:

```c
case 9:
    // Pump this much [uL]
    if (!motor.enable()) { error_type = ERROR_PUMP_MOTOR; ... break; }
    variable_value_float = Serial.parseFloat();
    motor.move(long(variable_value_float * steps_per_ml / 1000));
    moving_motor = true;
    break;
```

### f. Ack / response format

Because `write_acknowledge=True`, the host blocks in `_receive_acknowledge` (`device_arduino.py:632-696`) until a single ASCII line arrives. When the stepper finishes, the firmware emits `ACK_PUMPING 'k'` followed by `\n` (macro at `syringe_pump_nrg.ino:112`; `Serial.println('k')` paths at lines 430, 450, 468). A bad ack is `'n'` (`BAD_ACK_PUMPING`, line 113). The timeout is dynamically sized from `pumping_time(value) * 1.2` in `_write` above; an early stop sends the parameter's `stop_command` (`S21=RUN\n`), after which the device still emits the original `k`.

## Example 2 — `SetLightSourceIntensity` → lights-array proxy `intensity`

### a. Unit task

`SetLightSourceIntensity._execute` enables the array if needed, then writes the per-source intensity through the proxy's `__setitem__`.

`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/procedures/unit_tasks/reactor/light.py:41-60`:

```python
def _execute(cls, source, intensity):
    intensity = round(intensity)
    source_parent: LightArray = source.parent
    if intensity == 0:
        source["intensity"] = intensity       # <-- proxy write
        ...
    else:
        if source_parent["enable"] == "OFF":
            source_parent["enable"] = "ON"
        source["intensity"] = intensity       # <-- proxy write
```

### b. Device method

`LightSource` is a `ProxyArduinoDevice` attached to a `LightArray`. The proxy registers an `intensity` parameter whose `internal_id` is computed from the array slot:

`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/devices/nrg/light.py:75-88`:

```python
parameter = ArduinoParameter(
    name="intensity",
    access_level=ParameterAccess.RW,
    value_type=int,
    internal_id=base_variable_number,   # 20 + 10*array_index  (light.py:69-70)
)
parameter.min_value = 0
parameter.max_value = 100
self.add_parameter(parameter)
```

`base_variable_number` is computed in `array_arduino.py:43-55` as `_array_parameter_begin + _array_parameter_period * _array_index` — i.e. source 0 → id 20, source 1 → id 30, source 2 → id 40, source 3 → id 50.

### c. `BaseDevice.__setitem__`

Same generic path as Example 1 — `device.py:897-936`. The proxy does not override `_write`; it inherits `ArduinoDevice._write` via `ProxyArduinoDevice` (which forwards to the shared `_serial_iface` / `_serial_thread_lock` of its parent array), so the write reaches `ArduinoDevice._write` unchanged.

### d. Serial write

The same path as in Example 1 (`device_arduino.py:549-566` then `:768-771`). For `SetLightSourceIntensity.run(source_0, intensity=75)` the bytes on the wire are:

```
S20=75\n       -> 0x53 0x32 0x30 0x3D 0x37 0x35 0x0A
```

For source index 2 it would be `S40=75\n`. Intensity is declared `value_type=int`, so no `to_serial_string` override applies and plain `str(value)` is used (`_command_write` fallback branch).

### e. Firmware handler

The lights-array firmware splits variable ids: device-wide variables use the low ids, while ids `>= SERIAL_ARRAY_BEGIN` (20) are decoded to `(sub_id, local_var)` by `parse_array_command`. Local var 0 is the intensity setpoint.

`Devices/Photochemical Reactor Controller/Firmware/Lights_Array/Lights_Array.ino:374-403`:

```c
else if (command == 'S') {
    variable_number = Serial.parseInt();   // e.g. 20
    ...
    if (variable_number >= SERIAL_ARRAY_BEGIN) {
        parse_array_command(variable_number, sub_id);
        switch (variable_number) {                    // now local 0..4
            case 0:
                variable_value_int = Serial.parseInt();
                if (variable_value_int <= 100 && variable_value_int >= 0)
                    subdevices[sub_id].set_setpoint(variable_value_int);
                else { error_type = ERROR_SERIAL; ... }
                break;
```

### f. Ack / response format

The `intensity` parameter does not set `write_acknowledge`, so the host does not block for an ack. After the write, `ArduinoDevice._write` runs `_verify_error_register` in its `finally` clause (`device_arduino.py:778-779` and `:731-749`), which issues `R9\n` and parses the error-register reply in the `"type-value"` format (`ArduinoError.from_serial_string`, `device_arduino.py:184-197`). If `type != 0`, a matching `DeviceError` subclass is raised. A companion `enable` write on the parent array (also `Sx=y` with int 0/1 on id 3) uses the same silent pattern.

## General pattern (summary)

Every unit task is a thin classmethod-only orchestration layer over `device[name] = value` / `value = device[name]`. The reusable pipeline for a write is: unit task computes domain values → `BaseDevice.__setitem__` (`device.py:897`) validates and logs → concrete device `_write` injects policy (pre-checks, dynamic ack timeouts, cache invalidation) → `ArduinoDevice._write` (`device_arduino.py:751`) acquires the per-port `Lock`, flushes stray input, formats `Sx=y\n` with `_command_write` (`:549`), and writes ASCII at 9600 baud. If the parameter has `write_acknowledge`, `_receive_acknowledge` blocks for a single-line response matching `parameter.acknowledge_value` (e.g. `'k'`), otherwise `_verify_error_register` polls `Rerror\n` and raises typed exceptions on non-zero codes. Array devices reuse the same frame by encoding proxy index as `begin + period*index + local_id`, so the firmware switch-on-id in `Lights_Array.ino` decodes both device-wide and per-source variables with the same `Sx=y` grammar.
