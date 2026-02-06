# Phase Sensors Array usage

Control an Arduino device acting as an array of digital and analog ports.
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **max_digital** | READ ONLY | int | Maximum number of digital ports supported by array| | 
| **max_analog** | READ ONLY | int | Maximum number of analog ports supported by array| | 
| **error** | READ ONLY | GpioArrayError | Error register| | 
| **save** | WRITE ONLY | ArduinoValueRun | Save parameters as default|    value = RUN, | 
| **reset** | WRITE ONLY | ArduinoValueRun | Restore parameters to factory|    value = RUN, | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
| **array** | True | dict | Array devices must specify a list of subdevices via this setup option. |
