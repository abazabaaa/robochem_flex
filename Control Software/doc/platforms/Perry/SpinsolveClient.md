# Spinsolve NMR Client usage


    Connect to Spinsolve server via socket interface and access NMR device.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **sample** | WRITE ONLY | str | Set the sample name for the next measurement.| | 
| **solvent** | WRITE ONLY | str | Set the solvent for the next measurement.| | 
| **comment** | WRITE ONLY | str | Set the comment for the next measurement.| | 
| **data_folder** | WRITE ONLY | str | Set the folder where the next measurement will be stored.| | 
| **start** | WRITE ONLY | SpinsolveProtocol | Start an NMR protocol, such as measure or shim.| | 
| **run_script** | WRITE ONLY | str | Execute a spinsolve script.| | 
| **abort** | WRITE ONLY | ParameterValueRun | Abort an NMR protocol, such as measure or shim.| | 
| **available_protocols** | READ ONLY | str | Request a list of supported protocols.| | 
| **available_options_extended** | READ ONLY | str | Request a list of supported options for protocol xyz.| | 
| **available_options_shim** | READ ONLY | str | Request a list of supported options for protocol xyz.| | 
| **available_options_fluorine_hdec** | READ ONLY | str | Request a list of supported options for protocol xyz.| | 
| **available_options_fluorine** | READ ONLY | str | Request a list of supported options for protocol xyz.| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
