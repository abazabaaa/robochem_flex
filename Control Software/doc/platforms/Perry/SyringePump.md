# Syringe Pump usage


    Handles communication with the syringe pumps developed within the Noël Research Group, including up to two
    switch-valves. The main switch valve should be connected to the syringe, connecting it to a reservoir or flow
    system. The auxiliary switch valve is used within the liquid handler to control injection.
    The device is documented in detail at https://github.com/Noel-Research-Group/liquid_handler_cnc .

    Note: If present, the main switch valve should connect to the syringe on the 'C' (common) position,
        to the flow system on the 'ON' position, and to a solvent reservoir on the 'OFF' position.
        Connecting the switchvalve in a different way can result in malfunction during the initialization procedure.


    Usage:
        First, connect to the device with
        `self.open('COM4')`

        # todo add setup and initialize to all devices instructions.

        Then, change parameters of the device with
        `self['parameter_name'] = xyz`

        or read parameters with
        `xyz = self['parameter_name']`

        For a detailed list of available parameters, call
        `self.parameters()`

        For a human-readable list of parameters, call
        `print(self.usage())`

        Finally, close the device with
        `self.close()`
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **ID** | READ/WRITE | str | Device Identifier| | 
| **ack_pump** | READ/WRITE | ArduinoValueOnOff | Send acknowledge message after pump or zero move is complete| - OFF -> 0 , - ON -> 1 , | 
| **encoder** | READ/WRITE | ArduinoValueOnOff | Enable encoder wheel to detect blockages| - OFF -> 0 , - ON -> 1 , | 
| **valve_setpoint** | READ/WRITE | SwitchValveSetpointValue | Set switch-valve position| - OFF -> 1 , - ON -> 0 , | 
| **valve_actual** | READ ONLY | SwitchValvePositionValue | Switch-valve position read from sensor| - OFF -> 1 , - ON -> 0 , - ERROR -> 2 , | 
| **ack_valve** | READ/WRITE | ArduinoValueOnOff | Send acknowledge message after valve move is complete| - OFF -> 0 , - ON -> 1 , | 
| **aux_valve_setpoint** | READ/WRITE | SwitchValveSetpointValue | Set auxiliary switch-valve position| - OFF -> 1 , - ON -> 0 , | 
| **aux_valve_actual** | READ ONLY | SwitchValvePositionValue | Auxiliary switch-valve position read from sensor| - OFF -> 1 , - ON -> 0 , - ERROR -> 2 , | 
| **ack_aux_valve** | READ/WRITE | ArduinoValueOnOff | Send acknowledge message after auxiliary valve move is complete| - OFF -> 0 , - ON -> 1 , | 
| **diameter** | READ/WRITE | float | Syringe diameter| 0.1 <= value <= 100.0  mm, | 
| **steps_per_ml** | READ/WRITE | float | Motor steps per unit volume pumped| 10.0 <= value <= 10000000.0  1/mL, | 
| **flowrate** | READ/WRITE | float | Syringe pump flowrate| 0.0001 <= value <= 30.0  mL/min, | 
| **enable** | READ/WRITE | ArduinoValueOnOff | Enable syringe pump motor| - OFF -> 0 , - ON -> 1 , | 
| **stop** | WRITE ONLY | ArduinoValueRun | Stop syringe pumping or zeroing movements|    value = RUN, | 
| **pump** | WRITE ONLY | float | Pump volume now| -10000.000000000002 <= value <= 10000.000000000002  uL, | 
| **volume_left** | READ ONLY | float | Volume left since last pump command| -10000.000000000002 <= value <= 10000.000000000002  uL, | 
| **volume** | READ/WRITE | float | Volume of fluid within syringe| -10000.000000000002 <= value <= 10000.000000000002  uL, | 
| **zero** | WRITE ONLY | PumpZeroValue | Zero syringe pump in the specified direction| - EMPTY -> 0 , - FILL -> 1 , - IN_PLACE -> 2 , | 
| **flowrate_no_acc** | READ ONLY | float | Maximum pump flowrate achieved without acceleration| 0.0001 <= value <= 100.0  mL/min, | 
| **flowrate_acceleration** | READ ONLY | float | Acceleration for gradual increase of flowrate from 'flowrate_no_acc' to the target value| 0.0001 <= value <= 100.0  mL/min^2, | 
| **error** | READ ONLY | SyringePumpError | Error register| | 
| **save** | WRITE ONLY | ArduinoValueRun | Save parameters as default|    value = RUN, | 
| **reset** | WRITE ONLY | ArduinoValueRun | Restore parameters to factory|    value = RUN, | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
| **ignore_unexpected_data** | False | bool | Receiving unexpected data via serial communication is treated as a fatalerror by default, this option allows to only issue a warning when this happens. |
| **valve_setpoint_ignore_timeout** | False | bool | By default, timing out on parameter valve_setpoint raises an exception. If this option is set to True, only a warning is issued. |
| **valve_setpoint_ignore_badack** | False | bool | By default, receiving a bad acknowledge value from valve_setpoint raises an exception. If this option is set to True, only a warning is issued. |
| **aux_valve_setpoint_ignore_timeout** | False | bool | By default, timing out on parameter aux_valve_setpoint raises an exception. If this option is set to True, only a warning is issued. |
| **aux_valve_setpoint_ignore_badack** | False | bool | By default, receiving a bad acknowledge value from aux_valve_setpoint raises an exception. If this option is set to True, only a warning is issued. |
| **stop_ignore_timeout** | False | bool | By default, timing out on parameter stop raises an exception. If this option is set to True, only a warning is issued. |
| **stop_ignore_badack** | False | bool | By default, receiving a bad acknowledge value from stop raises an exception. If this option is set to True, only a warning is issued. |
| **pump_ignore_timeout** | False | bool | By default, timing out on parameter pump raises an exception. If this option is set to True, only a warning is issued. |
| **pump_ignore_badack** | False | bool | By default, receiving a bad acknowledge value from pump raises an exception. If this option is set to True, only a warning is issued. |
| **zero_ignore_timeout** | False | bool | By default, timing out on parameter zero raises an exception. If this option is set to True, only a warning is issued. |
| **zero_ignore_badack** | False | bool | By default, receiving a bad acknowledge value from zero raises an exception. If this option is set to True, only a warning is issued. |
| **syringe_volume** | True | str | Set the maximum volume of the syringe. Use a string including units (metric). |
| **syringe_stepperml** | False | str |float | Set the device motor-steps to volume conversion factor directly. |
| **syringe_diameter** | False | str | Set the syringe diameter to allow the device to know how much volume was pumped. Use a string including units (metric). |
| **max_flowrate** | True | float | Set the maximum flowrate allowed for the pump. |
| **valve** | False | bool | Enable access to the parameters controlling the main switchvalve. |
| **aux-valve** | False | bool | Enable access to the parameters controlling the auxiliary switchvalve. |
| **encoder** | False | bool | Disable encoder wheel feedback on the motor. |
| **reservoir_valve_position** | False | str | Sets the valve position for pumping to the reservoir (the opposite will be used to pump to the flow system). |
