# Chromtroller HPLC Client usage


    Connect to Chromtroller server via socket interface and access HPLC device.
    
## Parameters:

| Name | Access | Type | Description | Values |
|------|--------|------|-------------|--------|
| **run_info** | WRITE ONLY | dict | Info about the run to be logged by the server| | 
| **analysis_parameters** | WRITE ONLY | HPLCProcessingSettings | Settings to use for automatic chromatogram processing| | 
| **rt_target** | WRITE ONLY | float | Sets the target retention time for automatic chromatogram processing| | 
| **rt_tolerance** | WRITE ONLY | float | Sets the allowed tolerance in rt for automatic chromatogram processing| | 
| **valve_position** | WRITE ONLY | ValvePosValue | Set valve position. Either 'FILL' or 'INJECT'| - FILL -> FILL, - INJECT -> INJECT, | 
| **acquisition** | WRITE ONLY | ParameterValueRunHplc | Starts the acquisition procedure to analyse a sample stored in the sample loop| | 
| **analysis_result** | READ ONLY | dict | Starts the analysis procedure and returns the results| | 

## Setup options:
| Name | Required | Type | Description |
|------|----------|------|-------------|
