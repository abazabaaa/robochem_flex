# 09 — Arduino Firmware Protocol Reference

This report documents the low-level serial protocol exposed by every Arduino-based controller in `robochem_flex`. The host side (`device_arduino.py`) is covered elsewhere; the focus here is the firmware side.

## 1. Canonical Protocol

Every controller (except the Cartesian Sampler auxiliary axis) implements an identical ASCII, line-oriented protocol over USB-Serial at **9600 baud, 8N1**.

| Element | Specification |
|---|---|
| Baud rate | 9600 (`Serial.begin(9600)`) |
| Framing | ASCII; `\n` (LF) terminates responses. Commands are parsed char-by-char via `Serial.parseInt()`/`parseFloat()` and do not strictly require a terminator, but the host sends `\n`. |
| Write command | `Sx=y` — set variable `x` (int) to value `y` (int/float/string). The `=` is consumed by a bare `Serial.read()` between the index and value. |
| Read command | `Rx` — read variable `x`; firmware replies with a single `\n`-terminated value. |
| Error register format | `<type>-<value>\n` (see per-device error tables). Reading it resets the register. |
| Unknown variable | Silently sets the error register to `type=1, value=<variable>`; no line is emitted. |
| Asynchronous events | Controllers may emit unsolicited lines (valve/pump ACKs, phase-sensor state-change messages). The host must tolerate this. |
| Reserved index | `0` is never valid — `Serial.parseInt()` returns `0` on parse failure. |

Representative command parser (Syringe Pump, `syringe_pump_nrg.ino:225-235`):

```cpp
while (Serial.available() > 0) {
    char command = Serial.read();
    int variable_number = 0;
    if (command == 'R') {
        variable_number = Serial.parseInt();
        switch (variable_number) { /* ... per-variable readers ... */ }
    } else if (command == 'S') {
        variable_number = Serial.parseInt();
        Serial.read();              // discards '='
        /* Serial.parseInt/parseFloat reads the value */
    }
}
```

All five main-loop controllers follow this pattern. Array-style variables (indices ≥ `SERIAL_ARRAY_BEGIN`) are decoded with `parse_array_command()`: `sub_id = (idx - BASE) / PERIOD`, `sub_idx = (idx - BASE) % PERIOD`.

## 2. Per-Controller Reference

### 2.1 Syringe Pump (`syringe_pump_nrg.ino`)

Source: `/home/user/robochem_flex/Devices/Syringe Pump/Firmware/syringe_pump_nrg/syringe_pump_nrg.ino`.

| # | RW | Type | Meaning |
|---|---|---|---|
| 1 | RW | str[20] | Device identifier |
| 2 | RW | int[0-1] | Pump ACK enable (default 1) |
| 3 | RW | int[0-1] | Main valve setpoint (Runze 2-way) |
| 4 | RO | int[0-2] | Main valve position (2 = error) |
| 5 | RW | float | Syringe diameter [mm] |
| 6 | RW | float | Steps per mL [1/mL] (overridden by writing 5) |
| 7 | RW | float | Flow rate [mL/min] |
| 8 | RW | int[0-1] | Stepper driver enable |
| 9 | W  | float | Pump this volume now [µL] (signed) |
| 10 | RO | float | Volume remaining on current pump [µL] |
| 11 | RW | float | Volume currently in syringe [µL] |
| 12 | W  | int[0-2] | Zero syringe (0 empty / 1 full / 2 software-only) |
| 13 | RO | `t-v` | Error register (resets on read) |
| 14 | W  | — | Save defaults to EEPROM |
| 15 | W  | — | Factory reset |
| 16 | RW | int[0-1] | Auxiliary valve setpoint |
| 17 | RO | int[0-2] | Auxiliary valve position |
| 18 | RW | int[0-1] | Encoder wheel enable (line 294, gated on `WITH_ENCODER`) |
| 19 | RW | int[0-1] | Main-valve ACK enable |
| 20 | RW | int[0-1] | Aux-valve ACK enable |
| 21 | W  | — | Stop current movement |
| 22 | RO | float | Max no-accel flow [mL/min] |
| 23 | RO | float | Acceleration [mL/min²] |
| 24 | RW | int | Steps since last encoder interrupt (diagnostic) |

Async ACK characters (line 112-115): `'k'` = pump done, `'n'` = pump failed, `'v'` = main valve done, `'a'` = aux valve done.

Error codes (line 117-121): `0` none; `1` serial (value = bad var #); `2` valve; `3` motor (value is `MOTOR_ERROR_*` from `AsyncStepper.h:40-45`: `1` endstop+, `2` endstop–, `3` delay range, `4` disabled, `5` blocked/stalled).

**`loop()`** (line 569-610): Calls `parse_serial()` continuously and polls `valve.ready()`, `valve_aux.ready()`, and `motor.ready()` — emitting the corresponding ACK character when each completes. Motor errors raised during the move are folded into the error register and trigger `'n'` instead of `'k'`.

### 2.2 Photochemical Reactor Controller (`Lights_Array.ino`)

Source: `/home/user/robochem_flex/Devices/Photochemical Reactor Controller/Firmware/Lights_Array/Lights_Array.ino`.

Device-wide variables (`SERIAL_ARRAY_BEGIN = 20`):

| # | RW | Type | Meaning |
|---|---|---|---|
| 1 | RW | str[20] | Device identifier |
| 2 | RO | int | Number of subdevices (= 4) |
| 3 | RW | int[0-1] | SSR power-enable for all lights |
| 4 | RO | int | Measured supply current [mA] |
| 5 | RW | int[0-1024] | Current-sense offset |
| 6 | RW | int | Current-sense proportional factor |
| 7 | RO | float | Measured supply voltage [V] |
| 8 | RW | float | Voltage-sense proportional factor |
| 9 | RO | `t-v` | Error register (resets on read) |
| 10 | W | — | Save defaults |
| 11 | W | — | Factory reset |

Array variables, one set per light (`idx = 20 + 10·sub_id + i`, `sub_id ∈ 0..3`):

| i | RW | Type | Meaning |
|---|---|---|---|
| 0 | RW | int[0-100] | Intensity setpoint [%] |
| 1 | RW | float | Calibration (% → PWM raw) |
| 2 | RW | float | PID P constant |
| 3 | RW | float | PID I constant |
| 4 | RW | float | PID D constant |

Error codes (line 65-67): `0` none; `1` serial; `2` analog-out subsystem (see `Analog_out.h`, constant `ANALOGOUT_ERROR_OK = 0`).

**`loop()`** (line 521-527): tight `parse_serial()` loop. Closed-loop control is done in the `TIMER1_COMPA` ISR (line 533-540) every 100 ms, which calls `subdevices[i].update()` to drive each PID.

### 2.3 Phase Sensor Controller (`Phase_Sensor_Array.ino`)

Source: `/home/user/robochem_flex/Devices/Phase Sensor Controller/Firmware/Phase_Sensor_Array/Phase_Sensor_Array.ino`.

Device-wide variables (`SERIAL_ARRAY_BEGIN = 10`):

| # | RW | Type | Meaning |
|---|---|---|---|
| 1 | RW | str[20] | Device identifier |
| 2 | RO | int | Number of sensors (= 4) |
| 3 | RO | digits+`;` | Read all sensors (e.g. `1123;`) |
| 4 | RO | `t-v` | Error register (resets on read) |
| 5 | W | — | Save defaults |
| 6 | W | — | Factory reset |

Array variables per sensor (`idx = 10 + 10·sensor_id + i`):

| i | RW | Type | Meaning |
|---|---|---|---|
| 0 | RO | int[0-3] | Phase (0 err / 1 clear / 2 opaque / 3 gas — `OCB350.h:17-23`) |
| 1 | RO | int[0-1023] | Raw analog value |
| 2 | RW | int[0-2] | Monitor mode: `NO`/`ONCE`/`ALWAYS` |
| 3 | W | — | Run hardware calibration (tube must be dry) |

Error codes: `0` none; `1` serial.

Async message format: `<sensor_id>=<phase>\n` emitted from `loop()` when `sensors[i].has_changed()` returns true.

**`loop()`** (line 324-338): `parse_serial()` plus a per-sensor change-detection poll that calls `has_changed()` (which respects the `OCB_MIN_TIME` debounce from `OCB350.h:12`) and publishes state-change events.

### 2.4 General Purpose Controller (`GPIO_Array.ino`)

Source: `/home/user/robochem_flex/Devices/General Purpose Controller/Firmware/GPIO_Array/GPIO_Array.ino`.

Device-wide variables (`SERIAL_ARRAY_BEGIN = 10`):

| # | RW | Type | Meaning |
|---|---|---|---|
| 1 | RW | str[20] | Device identifier |
| 2 | RO | int | Number of GPIOs (= 10, pins D2–D12, skipping D13 LED) |
| 3 | RO | int | Number of analog inputs (= 6, A0–A5) |
| 4 | RO | `t-v` | Error register (resets on read) |
| 5 | W | — | Save defaults |
| 6 | W | — | Factory reset |

Array — digital pins (`gpio_id = pin − 2`, indices `10..109`):

| i | RW | Type | Meaning |
|---|---|---|---|
| 0 | RW | int[0-2] | Pin mode: output (0) / input (1) / PWM (2) |
| 1 | RO | int[0-1] | Digital read |
| 2 | RW | int[0-1] | Digital write (read returns last setvalue) |
| 3 | RW | int[0-255] | PWM duty (PWM-capable pins only) |

Array — analog inputs (`gpio_id = analog_pin + 11`, indices `120..170`):

| i | RW | Type | Meaning |
|---|---|---|---|
| 0 | RO | int[0-1023] | `analogRead()` |

Error codes (line 63-65, plus `GPIO.h:12-19`): `0` none; `1` serial; `2` pin error where value ∈ {`1` no PWM, `2` bad digital value, `3` bad PWM value, `4` write-to-input, `5` read-from-output, `6` invalid mode, `7` wrong-mode-for-op}.

**`loop()`** (line 352-358): a pure `parse_serial()` loop — the controller is completely reactive.

### 2.5 Cartesian Sampler — Auxiliary Axis (`auxiliary_axis.ino`)

Source: `/home/user/robochem_flex/Devices/Cartesian Sampler/Firmware/auxiliary_axis/auxiliary_axis.ino`.

This controller does **not** use the `Sx=y`/`Rx` protocol. It has no Serial interface. It drives a servo on `D9` between `POSITION_UP = 150` and `POSITION_DOWN = 75` (lines 3-5) based on a single digital input pin (`D2`). The LED on `D13` mirrors the state. Motion is rate-limited by a `DELAY = 5` ms per-degree step (lines 36-45).

Interface: `D2 HIGH` → axis drops; `D2 LOW` → axis rises. This is typically driven by a GPIO pin on the main Cartesian-sampler controller.

## 3. Device Identity (`Device_id.h`/`.cpp`)

Every controller ships an identical `Device_id` module (e.g. `/home/user/robochem_flex/Devices/Syringe Pump/Firmware/syringe_pump_nrg/Device_id.h:10-15`):

| Symbol | Definition |
|---|---|
| `DEVICE_ID_SIZE` | 20 (bytes) |
| `DEVICE_ID_CHAR_TIMEOUT` | 20 ms per char |
| `char device_id[20]` | Global buffer (`Device_id.cpp:11`) |

**Startup:** `setup()` calls `load_defaults()`, which reads an EEPROM check byte (`DEFAULT_CHECK = 42`). If present, `eeprom_get_id()` (`Device_id.cpp:37-47`) loads the stored name; otherwise `factory_reset()` writes a hard-coded default:

| Controller | Default ID |
|---|---|
| Syringe Pump | `SYRINGE_PUMP_0` |
| Photochemical Reactor | `LIGHTS_0` |
| Phase Sensor | `PHASE_ARRAY_0` |
| General Purpose | `GPIO_0` |

**Read:** `R1` prints `device_id\n`. **Write:** `S1=` followed by up to 20 chars terminated by `\n`, consumed by `serial_read_id()` (`Device_id.cpp:49-70`). The host uses this string for device discovery/matching.

## 4. Safety Interlocks in Firmware

The main safety-critical device is the syringe pump. Its interlocks live in `AsyncStepper.cpp`/`.h`:

| Mechanism | Defined in | Behaviour |
|---|---|---|
| Endstop pins | `AsyncStepper.h:41-42`, `syringe_pump_nrg.ino:108-109` | Two NO endstops (D8 neg, D9 pos). Active LOW. Unexpected activation → `MOTOR_ERROR_ENDSTOP_POS/NEG`. |
| Endstop debounce | `MAX_STEPS_SINCE_ENDSTOP = 6` (`AsyncStepper.h:24`) | A stop is only committed after 6 consecutive stepped reads show the endstop closed (`check_endstop()` at `AsyncStepper.cpp:336`). |
| Zero on hit | `endstop_stop()` (`AsyncStepper.cpp:321`) | Halts motion and zeros the absolute-position counter when an endstop closes during zeroing. |
| Stall detection (encoder) | `WITH_ENCODER`, `MAX_STEPS_SINCE_INT = 75` (`AsyncStepper.h:18,23`) | With the encoder wheel on D2 (INT0), every encoder pulse resets `steps_since_int`. If it exceeds ±75 the motor is considered blocked → `MOTOR_ERROR_BLOCKED`. Can be disabled at runtime via variable 18. |
| Step-delay clamp | `MIN_DELAY_US = 30`, `MAX_DELAY_US = 4_194_240` (`AsyncStepper.h:21-22`) | `set_step_delay()` returns `MOTOR_ERROR_DELAY_RANGE` outside this window, preventing ultra-fast commutation or stalls. |
| Disabled-driver guard | `MOTOR_ERROR_DISABLED` (lines 424-433, 443-453 of `.ino`) | `S9` (pump) and `S12` (zero, unless "software-only") reject the command and emit `'n'` if the driver is disabled. |
| Emergency stop | `S21` (line 519-530) | Host-driven hard-stop that aborts any pump/zero move in place and still returns the appropriate ACK. |
| EEPROM guard | `DEFAULT_CHECK = 42` byte | On boot, a corrupted/first-run EEPROM triggers `factory_reset()` instead of loading garbage calibration. Present in all four full controllers. |

Other controllers have lighter interlocks: the Lights controller tracks an `AnalogOut::error` static and returns it under variable 9, and the GPIO controller raises `GPIO_ERROR_WRITE_TO_INPUT`/`READ_FROM_OUTPUT`/`WRONG_MODE` on misuse (`GPIO.h:12-19`). None of them include a hardware watchdog — timing safety relies on the host refreshing setpoints and the blocking nature of the parser loops.
