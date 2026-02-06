"""
File: __init__.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: Package header for the device control units of the omniplatypus.
    General utilities are found in modules directly below 'devices', while specific devices are found in
    packages named after the manufacturer. Self-developed devices are found under 'nrg'.
    General prototypes for devices are in the 'base' package.

    Note on units: In many cases, parameters and property store values and their units separately. Units are stored as
    strings, with or without leading space (depending on whether a space is expected between values and units).
    However, this module does not support all units and combinations. For the most part, the SI system is used.
    SI prefixes are well supported.
    When values are stored directly, without specifying units, the following assumptions should be made (specific
    function or class documentation overrides this, always refer to that):
    - volume [uL]
    - time [S]
    - velocity [mm/min]
    - flowrate [mL/min]
    - temperature [°C]
    - concentration [M]
    - relative amount [eq]
"""
