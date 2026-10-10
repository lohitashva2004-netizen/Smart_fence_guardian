# Signal model (assumptions)

All amplitudes are per-unit (pu) of the nominal 50 Hz peak seen by the sensor. These shapes are **assumed**, chosen to
capture the qualitative physics, and must be replaced by measured captures before any real-world claim.

| Class | Model | Parameter ranges |
|---|---|---|
| background | 50 Hz + 3rd/5th harmonics; 30 % of windows contain a small impulsive-noise burst | burst 0.05-0.4 pu |
| motor start | 50 Hz inrush with slow rise and decay | 1-3.5 pu, rise 0.3-1.5 ms, decay 15-80 ms |
| lightning / surge | double exponential, optional ringing | 2-12 pu (log-uniform), rise tau 1-12 us, decay tau 30-150 us |
| switching | damped sinusoid with random phase | 0.4-3 pu, 1-12 kHz, decay 0.2-1.5 ms |
| **fence energizer** | double exponential, optional ringing | 0.4-3.5 pu (log-uniform), rise tau 6-25 us, decay tau 90-350 us, ~1 pulse/s |

Front end: 200 kS/s, 12-bit, +/-10 pu full scale, white Gaussian noise (default 35 dB below the mains peak), amplitude
attenuation `exp(-d / 20 km)` along the feeder. Propagation velocity 2.9e8 m/s (overhead line).

**Why the ranges overlap on purpose:** fence and lightning pulses share a waveform family. If they were cleanly separated,
the classifier results would say more about the simulator than about the method.

**The biggest unknown:** the shape of a fence-energizer pulse as it appears on the *grid side*, behind the energizer's own
power stage and the consumer's wiring, is not characterised here. Measuring it is the first thing to do with real hardware.
