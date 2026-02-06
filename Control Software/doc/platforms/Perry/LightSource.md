# Light Source usage


    Control the modulation signal sent to a single light source.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **intensity** | READ/WRITE | int | Set the light intensity as a percentage of the source capacity| 0 <= value <= 100 %, | 
| **calibration** | READ/WRITE | float | Proportional calibration value used for generating the signal| | 
| **P** | READ/WRITE | float | Proportional component for the light control PID signal generator| | 
| **I** | READ/WRITE | float | Integral component for the light control PID signal generator| | 
| **D** | READ/WRITE | float | Derivative component for the light control PID signal generator| | 
| **error** | READ ONLY | LightArrayError | Error register| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
