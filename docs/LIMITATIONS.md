# Limitations and threats to validity

1. **Simulation only.** No real feeder data was used. Every performance figure is conditional on the assumed signal model
   (`docs/SIGNAL_MODEL.md`). The results demonstrate that the *pipeline works if the assumed signatures hold*.
2. **Grid-side signature is unverified.** Whether a fence pulse is visible, and how distinct it is, at the distribution
   transformer is the central physical assumption of the whole concept.
3. **Clock synchronisation limits localisation.** Position error is roughly `v * sigma_sync / sqrt(2)`: about 200 m per
   microsecond of node-to-node timing error. The sensors therefore need GPS-disciplined timestamps (or an equivalent
   common time reference). Localisation is not "GPS-free" in this design.
4. **Fence must lie between sensors.** A source beyond the outermost sensor is unobservable from arrival-time differences
   (shown in `results/figures/09_span_ambiguity.png`).
5. **Simplified line model.** Single straight feeder, constant velocity, simple exponential attenuation. Real networks have
   branches, transformers, cables with different velocities and frequency-dependent dispersion that distorts pulse edges.
6. **Other periodic pulse sources are not modelled.** Welders, thyristor drives, spark-gap devices or a legal, licensed
   fence energizer would all look "periodic". Deployment needs a registry of authorised installations.
7. **Legal and ethical context.** The system flags a signature; a human decision is needed before any enforcement action.
   Many fences are legal; the goal is detecting *unauthorised grid connection*, which requires context beyond waveforms.
8. **Small simulated sample.** False-alarm results are "zero observed in N simulated hours", which is not a guarantee of
   zero in the field (see the upper bound stated in the README).
