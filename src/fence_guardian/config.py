"""Central configuration. All physical parameters here are *modelling assumptions*; see docs/SIGNAL_MODEL.md."""

SEED = 42

# Event classes the edge classifier distinguishes.
CLASSES = ["background", "motor_start", "lightning", "switching", "energizer"]
ENERGIZER = CLASSES.index("energizer")

# Sensor front-end
FS_HZ = 200_000          # sampling rate (ESP32-class ADC realistic upper range)
WINDOW_S = 0.010         # capture window around a trigger
ADC_BITS = 12
ADC_FULL_SCALE_PU = 10.0  # +/- full scale, in per-unit of the nominal 50 Hz current/voltage peak
LINE_HZ = 50.0

# Pre-trigger layout inside every capture window
NOISE_REGION_S = (0.0002, 0.0012)   # event-free region used to estimate the noise floor
ONSET_RANGE_S = (0.0025, 0.0060)    # where the event starts inside the window

# Propagation / feeder model
PROP_VELOCITY_M_S = 2.9e8           # ~0.97 c, overhead line
ATTEN_LENGTH_M = 20_000.0           # amplitude falls as exp(-d / L)
FEEDER_LENGTH_M = 10_000.0
