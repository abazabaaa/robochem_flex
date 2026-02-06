# Phase Sensors Array usage


    Control an array of phase sensors.

    Attributes:
        default_timeout: float  Default time in seconds before the wait parameter or function returns.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **max_sensors** | READ ONLY | int | Maximum number of sensors supported by array| | 
| **read_all** | READ ONLY | PhaseList | Read all sensors within array| | 
| **error** | READ ONLY | PhaseSensorArrayError | Error register| | 
| **save** | WRITE ONLY | ArduinoValueRun | Save parameters as default|    value = RUN, | 
| **reset** | WRITE ONLY | ArduinoValueRun | Restore parameters to factory|    value = RUN, | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
| **array** | True | dict | Array devices must specify a list of subdevices via this setup option. |
