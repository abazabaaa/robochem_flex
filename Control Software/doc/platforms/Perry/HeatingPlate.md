# IKA Heating Plate usage


## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **name** | READ ONLY | str | Read device name| | 
| **actual_sensor_temperature** | READ ONLY | float | Read actual external sensor value| | 
| **actual_hotplate_temperature** | READ ONLY | float | Read actual hotplate value| | 
| **actual_stirring_speed** | READ ONLY | float | Read stirring speed value | | 
| **read_rated_temperature** | READ ONLY | str | Read rated temperature value| | 
| **read_rated_safety_temperature** | READ ONLY | str | Read rated set safety temperature value| | 
| **read_rated_speed** | READ ONLY | float | Read rated speed value| | 
| **set_temperature** | WRITE ONLY | float | Adjust the set temperature value| 0.0 <= value <= 310.0 °C, | 
| **set_stirring** | WRITE ONLY | float | Adjust the set speed value| 0.0 <= value <= 1500.0 rpm, | 
| **start_heating** | WRITE ONLY | str | Start the heater| | 
| **stop_heating** | WRITE ONLY | str | Stop the heater| | 
| **start_stirring** | WRITE ONLY | str | Start the stirring| | 
| **stop_stirring** | WRITE ONLY | str | Stop the stirring| | 
| **reset** | READ ONLY | str | Switch to normal operating mode| | 
| **set_operating_mode** | READ/WRITE | str | Set operating mode| | 
| **safety_temperature_echo** | READ ONLY | float | Setting WD safety limit temperature with set value echo| | 
| **safety_speed_echo** | READ ONLY | float | Setting WD safety limit speed with set value echo| | 
| **watchdog_mode1** | READ ONLY | float | Watchdog mode 1:| | 
| **watchdog_mode2** | READ ONLY | float | Watchdog mode 2:| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
