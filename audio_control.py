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
            # Remove old claps (> 1.5s ago)
            self.claps = [t for t in self.claps if now - t < 1.5]
            
            # Check patterns
            # Single clap pattern? Hard to distinguish from noise instantaneously.
            # Usually we wait a bit to see if another clap comes.
            # But the requirement is "Single clap = relaxing" and "Double clap = party".
            # If we detect 1 clap and no more for (say) 0.5s, trigger single.
            # If we detect 2 claps, trigger double immediately.
            
            if len(self.claps) == 2:
                self.light_controller.set_mode("Party")
                self.claps = [] # Reset
            elif len(self.claps) == 1:
                # Check if it has been hanging there for 0.6s
                if now - self.claps[0] > 0.6:
                    self.light_controller.set_mode("Relaxing")
                    self.claps = []

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
