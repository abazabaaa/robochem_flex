# Raman Spectrometer usage


    Handles communication with RamaBerry device.
    Uses the RamaBerry library, wrapping the instrument class to implement the BaseDevice interface.

    Usage:
        First, connect to the device with
        `self.open()'

        then set the spectrometer parameters with:
        `self['parameter_name'] = xyz`

        or read the spectrometer parameters with:
        `xyz = self['parameter_name']`

        For a detailed list of available parameters, call
        `self.parameters()`

        For a human-readable list of parameters, call
        `print(self.usage())`
        (amongst parameters there is data as well so be careful)


        Finally, close the device with
        `self.close()`
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **integration_time** | WRITE ONLY | float | integration time of the spectrometer(ms)| 100.0 <= value <= 50000.0 ms, | 
| **n_averages** | READ/WRITE | int | number of averages| 1 <= value <= 100 int, | 
| **delay** | READ/WRITE | float | delay between measurements (ms)| 0.0 <= value <= 500000.0 ms, | 
| **n_scans** | READ/WRITE | int | number of scans| 1 <= value <= 1000 int, | 
| **save_as** | READ/WRITE | DataPollerStyle | return type defines shape of return data values (server_kinetic, server_single)| - SERVER_SINGLE -> SERVER_SINGLE , - SERVER_KINETIC -> SERVER_KINETIC , | 
| **dark_correction** | READ/WRITE | bool | dark correction pre spectrum| | 
| **data** | READ ONLY | DataFrame | spectral data| | 
| **start_acq** | READ/WRITE | bool | start acquisition, set to true to start, will go back to false when acq is started| | 
| **stop_acq** | READ/WRITE | bool | stop acquisition, set to true to stop, will go back to false when acq is stopped| | 
| **save_moniker** | READ/WRITE | str | Save name addition, things are saved as Moniker_YYYY_MM_DD_HHMMSSsss in the folderrobochem_spectra/YYYYMMDD| | 
| **save_path** | READ/WRITE | str | Data folder to save stuff on, default is in the cloud place| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
