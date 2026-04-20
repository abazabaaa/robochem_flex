# Reports Index

This directory contains ten cross-cutting reports that dissect the `robochem_flex` self-driving-lab codebase from the wire protocol up to the Bayesian-optimisation loop and the analytics feedback path. Each report is self-contained but cross-references the others; together they answer the question "how does this code control a robot, and how does it close the loop with machine learning?". File/line citations point into `Control Software/OmniPlatypus/…`, `Control Software/backend/…`, `Control Software/Lamas/…`, and `Devices/*/Firmware/*.ino`. Start here, then read in one of the two orders below. For a single-document synthesis, see `../REPO_MAP.md`.

## Contents

| # | Title | Purpose |
|---|---|---|
| 01 | Hardware Communication Layer | Dissects `BaseDevice`, the `ArduinoDevice`/`ModbusDevice`/`SocketDevice` transports, and the `ArrayDevice`/`ProxyDevice` pattern that lets many sub-units share one serial port. |
| 02 | Platform & Device Configuration | Explains `platform_config.json`, `Platform.build()`, `known_devices.json` serial-ID discovery, and `SampleHolder` vial-rack geometry. |
| 03 | Unit Task to Wire Trace | Follows `PumpVolume` and `SetLightSourceIntensity` all the way from Python down to the literal ASCII bytes on the serial wire and the matching firmware `case` handler. |
| 04 | Experiment Orchestration Lifecycle | Documents `BaseExperiment`'s threads, five queues, `RunResult`, and the `ChemicalReaction`/`PhotochemicalReaction`/`ThermochemicalReaction` subclasses. |
| 05 | ML - Hardware Closed Loop | Traces BoTorch backends, `ToandFromMachine` tensor/recipe translation, and the queue boundary between the ML thread and the experiment thread. |
| 06 | Sample Prep & Liquid Handling | Covers `StockSolutionDF`/`VialDF`, the `GenerateComposition` linear-programming recipe solver, and the pump/sampler unit tasks. |
| 07 | Analytics & Measurement Feedback | Walks raw spectra through Alpaca, Vicuna, Guanaco, and Lama; contrasts NMR/HPLC/Raman/UV-Vis analyser classes; defines the `result_metrics`/`pass` contract. |
| 08 | End-to-End CS1 Walkthrough | A single Bayesian iteration of the `Examples/CS1` photoredox campaign on platform Perry, from Start button to refitted GP posterior. |
| 09 | Arduino Firmware Protocol Reference | Per-device variable tables, the `Sx=y`/`Rx` line protocol at 9600 baud, ACK characters, and firmware safety interlocks. |
| 10 | Safety, Errors & Recovery | Error-class taxonomy, the three-tier retry/restart policy, known gaps including the missing hardware emergency-stop. |

## Read in this order

- **To understand how the code controls a robot** (bottom-up mechanics): start with **02** (platform configuration), then **01** (the `BaseDevice` abstraction), then **09** (what the firmware actually expects on the wire), then **03** (a worked wire-trace), then **08** (it all comes together).
- **To understand the ML loop** (top-down chemistry and learning): start with **05** (the optimizer and the queue boundary), then **06** (how a tensor becomes a real slug), then **07** (how a spectrum becomes a scalar), then **04** (the orchestrator that threads the two together), then **08** (a concrete campaign iteration).

Reports 09 and 10 are reference material — consult them on demand when the other reports cite specific firmware variables or error classes.
