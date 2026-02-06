# Solenoid Valve usage


    Control a solenoid valve via a DigitalPort.
    This is functionally equivalent to a DigitalPort which comes pre-configured as a digital output to drive a solenoid
    valve using an external power stage (mosfet, relay, ecc...).
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **mode** | READ/WRITE | PortModeValue | Set the port mode| - OUTPUT -> 0 , - INPUT -> 1 , - PWM -> 2 , | 
| **read** | READ ONLY | DigitalPortValue | Read the port value as a digital input| - OFF -> 0 , - ON -> 1 , | 
| **write** | READ/WRITE | DigitalPortValue | Write the port value as a digital output, reads set-value| - OFF -> 0 , - ON -> 1 , | 
| **pwm** | READ/WRITE | int | Write the port value as a PWM value, reads set-value| 0 <= value <= 255 , | 
| **error** | READ ONLY | GpioArrayError | Error register| | 
| **valve** | READ/WRITE | SolenoidValveValue | Actuate the solenoid valve| - CLOSE -> 0 , - OPEN -> 1 , | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
