# Phase Sensor usage


    Read and monitor the state of a single phase sensor. These are grouped in arrays and cannot be instantiated as
    stand alone.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **phase** | READ ONLY | PhaseValue | Read the phase at the sensor digitally| - ERROR -> 0 , - CLEAR_LIQUID -> 1 , - OPAQUE_LIQUID -> 2 , - GAS -> 3 , | 
| **analog** | READ ONLY | int | Read the phase at the sensor analogically| 0 <= value <= 1023 , | 
| **monitor** | READ/WRITE | PhaseMonitoringValue | Generate messages upon phase change| - NEVER -> 0 , - ONCE -> 1 , - ALWAYS -> 2 , | 
| **calibrate** | WRITE ONLY | ArduinoValueRun | Calibrate sensor|    value = RUN, | 
| **error** | READ ONLY | PhaseSensorArrayError | Error register| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
