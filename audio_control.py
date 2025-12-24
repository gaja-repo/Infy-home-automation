import pyaudio
import numpy as np
import time
import threading

class AudioController:
    def __init__(self, light_controller):
        self.light_controller = light_controller
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.threshold = 1000  # Adjust based on mic sensitivity
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.claps = []
        self.lock = threading.Lock()

    def start(self):
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        self.running = True
        self.thread = threading.Thread(target=self.listen)
        self.thread.start()

    def listen(self):
        print("[AUDIO] Listening for claps...")
        while self.running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                data_int = np.frombuffer(data, dtype=np.int16)
                peak = np.average(np.abs(data_int))

                if peak > self.threshold:
                    now = time.time()
                    with self.lock:
                        # Debounce: ignore claps too close together (< 0.1s)
                        if not self.claps or now - self.claps[-1] > 0.1:
                            self.claps.append(now)
                            # print(f"[AUDIO] Clap detected! (Peak: {peak})")
                            
            except Exception as e:
                print(f"[AUDIO] Error: {e}")
                
            self.process_claps()

    def process_claps(self):
        now = time.time()
        with self.lock:
            # Remove old claps (> 2.0s ago) - extended for triple clap
            self.claps = [t for t in self.claps if now - t < 2.0]
            
            # Check patterns:
            # Single clap = Normal mode
            # Double clap = Relaxing mode
            # Triple clap = Party mode
            
            if len(self.claps) >= 3:
                # Triple clap detected - Party mode
                self.light_controller.set_mode("Party")
                print("[AUDIO] Triple clap detected - Party Mode!")
                self.claps = []  # Reset
            elif len(self.claps) == 2:
                # Wait a bit to see if a third clap comes
                if now - self.claps[-1] > 0.5:
                    # No third clap, it's a double clap - Relaxing mode
                    self.light_controller.set_mode("Relaxing")
                    print("[AUDIO] Double clap detected - Relaxing Mode!")
                    self.claps = []  # Reset
            elif len(self.claps) == 1:
                # Wait to see if more claps come
                if now - self.claps[0] > 0.7:
                    # No more claps, it's a single clap - Normal mode
                    self.light_controller.set_mode("Normal")
                    print("[AUDIO] Single clap detected - Normal Mode!")
                    self.claps = []  # Reset

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
