# Lights Array usage


    Arduino device controlling several analog 10v outputs, each modulating a light source power output.
    The device equips current and voltage monitoring functions to measure electrical power of the sources.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **max_lights** | READ ONLY | int | Maximum number of light sources controlled by one array device| | 
| **enable** | READ/WRITE | ArduinoValueOnOff | Enable power source for all light sources connected| - OFF -> 0 , - ON -> 1 , | 
| **current** | READ ONLY | int | Total measured electrical current delivered to the light sources| 0 <= value <= 5000  mA, | 
| **current_q** | READ/WRITE | int | Zero offset calibration value for the measured current| 0 <= value <= 1024 , | 
| **current_m** | READ/WRITE | int | Proportional calibration factor for the measured current| | 
| **voltage** | READ ONLY | float | Measured voltage delivered to the light sources| | 
| **voltage_m** | READ/WRITE | float | Proportional calibration factor for the measured voltage| | 
| **error** | READ ONLY | LightArrayError | Error register| | 
| **save** | WRITE ONLY | ArduinoValueRun | Save parameters as default|    value = RUN, | 
| **reset** | WRITE ONLY | ArduinoValueRun | Restore parameters to factory|    value = RUN, | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
| **array** | True | dict | Array devices must specify a list of subdevices via this setup option. |
