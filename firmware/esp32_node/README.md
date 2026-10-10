# ESP32 sensor node (reference sketch)

**Status: design sketch, not validated on hardware.** It documents the timing architecture assumed by the simulation
and gives the team a starting point; it has not been compiled for a board or tested with real signals.

| Block | Approach | State |
|---|---|---|
| Pulse onset timestamp | analog peak detector + comparator -> GPIO interrupt -> 240 MHz cycle counter | sketched in `src/main.cpp` |
| Clock discipline | GPS 1PPS interrupt; event time = PPS second + cycle delta / measured cycles per second | sketched |
| Waveform capture | fast external ADC (the ESP32's internal ADC is too slow/noisy for 200 kS/s edge shapes) | TODO |
| Feature extraction | port of `src/fence_guardian/features.py` to C/C++ | TODO |
| Edge classifier | `src/fence_classifier.h`, a depth-6 tree **auto-generated** from the Python model | generated and verified against Python with gcc in `tests/test_c_export.py` |
| Uplink | JSON per `fence_guardian.messaging` over LoRa/GSM -> MQTT | schema done, transport TODO |

Regenerate the classifier header with `python scripts/run_experiments.py`.

**High-voltage safety:** this node must only ever see the grid through certified, isolated current/voltage transducers
(CT / PT) with surge protection. Never connect mains or HV directly to the ESP32 or comparator.
