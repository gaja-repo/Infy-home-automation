import pyaudio
import numpy as np
import time
import threading
from collections import deque

class AudioController:
    """Improved audio controller with robust clap detection."""
    
    def __init__(self, light_controller):
        self.light_controller = light_controller
        self.chunk = 2048  # Larger chunk for better analysis
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        # Adaptive threshold with ambient noise calibration
        self.base_threshold = 2000
        self.threshold_multiplier = 3.0
        self.ambient_level = 500
        self.dynamic_threshold = self.base_threshold
        
        # Clap detection parameters
        self.min_clap_interval = 0.12  # Min time between claps
        self.max_clap_interval = 0.6   # Max time between claps in a pattern
        self.pattern_timeout = 0.8     # Time to wait before finalizing pattern
        
        # Energy-based clap validation
        self.clap_min_energy = 5000
        self.clap_max_duration_samples = int(0.15 * self.rate)  # 150ms max
        
        # State
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.claps = []
        self.lock = threading.Lock()
        
        # Ambient noise tracking
        self.ambient_samples = deque(maxlen=50)
        self.is_calibrating = True
        self.calibration_frames = 0
        
        # Peak detection state
        self.last_peak_time = 0
        self.in_clap = False
        self.clap_start_time = 0
        
        # Pattern history for stability
        self.last_pattern_time = 0
        self.pattern_cooldown = 1.5  # Seconds between pattern triggers

    def start(self):
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.running = True
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        print("[AUDIO] Listening for claps... (calibrating ambient noise)")
        
        while self.running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                data_int = np.frombuffer(data, dtype=np.int16).astype(np.float64)
                
                # Calculate audio features
                peak = np.max(np.abs(data_int))
                rms = np.sqrt(np.mean(np.square(data_int)))
                
                # Calibrate ambient noise level
                if self.is_calibrating:
                    self.ambient_samples.append(rms)
                    self.calibration_frames += 1
                    if self.calibration_frames >= 30:
                        self.ambient_level = np.mean(self.ambient_samples) if self.ambient_samples else 500
                        self.dynamic_threshold = max(self.base_threshold, 
                                                    self.ambient_level * self.threshold_multiplier)
                        self.is_calibrating = False
                        print(f"[AUDIO] Calibrated. Ambient: {int(self.ambient_level)}, Threshold: {int(self.dynamic_threshold)}")
                else:
                    # Update ambient level slowly
                    if rms < self.dynamic_threshold * 0.5:
                        self.ambient_samples.append(rms)
                        self.ambient_level = np.mean(self.ambient_samples)
                        self.dynamic_threshold = max(self.base_threshold,
                                                    self.ambient_level * self.threshold_multiplier)
                
                # Detect clap
                now = time.time()
                
                if peak > self.dynamic_threshold:
                    # Check if this is a new clap (not continuation of previous)
                    if now - self.last_peak_time > self.min_clap_interval:
                        # Validate it's a clap (sharp attack, quick decay)
                        if self._is_valid_clap(data_int, peak):
                            with self.lock:
                                self.claps.append(now)
                                print(f"[AUDIO] Clap detected! (Peak: {int(peak)}, Count: {len(self.claps)})")
                            self.last_peak_time = now
                
                # Process clap patterns
                self.process_claps()
                
            except Exception as e:
                if self.running:
                    print(f"[AUDIO] Error: {e}")
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def _is_valid_clap(self, data, peak):
        """Validate that the sound is a clap (sharp attack, quick decay)."""
        # Find where the peak occurs
        peak_idx = np.argmax(np.abs(data))
        
        # Check if it's in the first half (sharp attack)
        if peak_idx > len(data) * 0.7:
            return False  # Peak too late, probably not a clap
        
        # Check decay - samples after peak should reduce quickly
        if peak_idx < len(data) - 100:
            post_peak = np.abs(data[peak_idx:peak_idx + 100])
            decay_ratio = np.mean(post_peak[-50:]) / (peak + 1)
            if decay_ratio > 0.5:
                return False  # Not enough decay, probably sustained noise
        
        return True

    def process_claps(self):
        now = time.time()
        
        with self.lock:
            # Remove old claps
            self.claps = [t for t in self.claps if now - t < 2.0]
            
            if len(self.claps) == 0:
                return
            
            # Check if we're in cooldown
            if now - self.last_pattern_time < self.pattern_cooldown:
                return
            
            # Check if pattern is complete (enough time since last clap)
            time_since_last = now - self.claps[-1]
            
            if time_since_last < self.pattern_timeout:
                return  # Still waiting for more claps
            
            # Validate clap intervals
            valid_pattern = True
            for i in range(1, len(self.claps)):
                interval = self.claps[i] - self.claps[i-1]
                if interval < self.min_clap_interval or interval > self.max_clap_interval:
                    valid_pattern = False
                    break
            
            if not valid_pattern:
                self.claps = []
                return
            
            # Determine pattern
            clap_count = len(self.claps)
            
            if clap_count >= 3:
                self.light_controller.set_mode("Party")
                print("[AUDIO] ✓ Triple clap confirmed - PARTY MODE!")
                self.last_pattern_time = now
                self.claps = []
            elif clap_count == 2:
                self.light_controller.set_mode("Relaxing")
                print("[AUDIO] ✓ Double clap confirmed - RELAXING MODE!")
                self.last_pattern_time = now
                self.claps = []
            elif clap_count == 1:
                self.light_controller.set_mode("Normal")
                print("[AUDIO] ✓ Single clap confirmed - NORMAL MODE!")
                self.last_pattern_time = now
                self.claps = []

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
