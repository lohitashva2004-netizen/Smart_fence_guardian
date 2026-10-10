/*
 * Smart Fence Guardian - sensor node reference sketch (ESP32, Arduino framework)
 *
 * STATUS: design sketch. Not compiled or tested on hardware. It shows the timing architecture the
 * simulation assumes; the analog front end, ADC capture and feature extraction are still TODO.
 *
 * Timing idea (why localisation can work on a microcontroller):
 *   - An analog peak detector + comparator turns the first fast edge of a pulse into a GPIO edge.
 *   - A GPIO interrupt latches the 240 MHz CPU cycle counter (~4 ns resolution) - far finer than any ADC sample period.
 *   - A GPS 1PPS interrupt latches the same counter once per second. An event time is then
 *       gps_second + (event_cycles - pps_cycles) / measured_cycles_per_second
 *     which also cancels crystal drift. Residual error is dominated by comparator walk and PPS jitter.
 */
#include <Arduino.h>
#include "fence_classifier.h"   // generated: python scripts/run_experiments.py

constexpr int PIN_TRIGGER = 25;   // comparator output, rising edge = pulse onset
constexpr int PIN_PPS     = 27;   // GPS 1PPS, rising edge
constexpr size_t QUEUE_LEN = 32;

struct RawEvent {
  uint32_t event_cycles;   // cycle counter at trigger
  uint32_t pps_cycles;     // cycle counter at the most recent PPS
  uint32_t pps_seq;        // which PPS that was
};

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t g_pps_cycles = 0, g_pps_seq = 0, g_cycles_per_s = 240000000UL;
volatile RawEvent g_queue[QUEUE_LEN];
volatile size_t g_head = 0, g_tail = 0;

void IRAM_ATTR onPps() {
  uint32_t now = ESP.getCycleCount();
  portENTER_CRITICAL_ISR(&mux);
  if (g_pps_seq > 0) g_cycles_per_s = now - g_pps_cycles;     // measured, so crystal drift cancels
  g_pps_cycles = now;
  g_pps_seq++;
  portEXIT_CRITICAL_ISR(&mux);
}

void IRAM_ATTR onTrigger() {
  uint32_t now = ESP.getCycleCount();
  size_t next = (g_head + 1) % QUEUE_LEN;
  if (next == g_tail) return;                                 // queue full: drop
  g_queue[g_head].event_cycles = now;
  g_queue[g_head].pps_cycles = g_pps_cycles;
  g_queue[g_head].pps_seq = g_pps_seq;
  g_head = next;
}

// TODO: parse NMEA (e.g. TinyGPS++) and return the UTC epoch second that the given PPS marks.
uint64_t gpsEpochSecondForPps(uint32_t /*pps_seq*/) { return 0; }

// TODO: capture a window from the ADC (external fast ADC over SPI/I2S) and compute the 13 features
// listed in src/fence_guardian/features.py. Until then the node only reports timestamps.
bool captureAndClassify(int* out_class) { (void)out_class; return false; }

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIGGER, INPUT);
  pinMode(PIN_PPS, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_PPS), onPps, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_TRIGGER), onTrigger, RISING);
}

void loop() {
  while (g_tail != g_head) {
    RawEvent e;
    portENTER_CRITICAL(&mux);
    e = {g_queue[g_tail].event_cycles, g_queue[g_tail].pps_cycles, g_queue[g_tail].pps_seq};
    uint32_t cps = g_cycles_per_s;
    portEXIT_CRITICAL(&mux);
    g_tail = (g_tail + 1) % QUEUE_LEN;

    double frac_s = (double)(uint32_t)(e.event_cycles - e.pps_cycles) / (double)cps;
    uint64_t t_ns = gpsEpochSecondForPps(e.pps_seq) * 1000000000ULL + (uint64_t)(frac_s * 1e9);

    int cls = -1;
    captureAndClassify(&cls);
    // Same schema as fence_guardian.messaging.EventMessage (publish over MQTT / LoRaWAN instead of Serial).
    Serial.printf("{\"schema\":1,\"node_id\":\"sub1\",\"seq\":%u,\"timestamp_ns\":%llu,\"edge_class\":%d}\n",
                  (unsigned)e.pps_seq, (unsigned long long)t_ns, cls);
  }
}
