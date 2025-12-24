import time

class LightController:
    """
    Simulates a home automation light controller with debouncing.
    In a real system, this would interface with hardware (Hue, Tuya, GPIO, etc.).
    """
    def __init__(self):
        self.is_on = False
        self.brightness = 50  # 0 to 100
        self.mode = "Normal"  # Normal, Relaxing, Party
        
        # Debouncing variables
        self.last_mode_change = 0
        self.mode_debounce_time = 0.5  # Seconds between mode changes
        self.last_toggle_time = 0
        self.toggle_debounce_time = 0.5  # Seconds between on/off toggles

    def turn_on(self):
        now = time.time()
        if not self.is_on and (now - self.last_toggle_time > self.toggle_debounce_time):
            self.is_on = True
            self.last_toggle_time = now
            print("[LIGHT] Turned ON")

    def turn_off(self):
        now = time.time()
        if self.is_on and (now - self.last_toggle_time > self.toggle_debounce_time):
            self.is_on = False
            self.last_toggle_time = now
            print("[LIGHT] Turned OFF")

    def set_brightness(self, level):
        """Sets brightness between 0 and 100."""
        # Brightness updates happen frequently (streaming), so no strict debounce
        # But we check state first
        if not self.is_on:
            # print("[LIGHT] Ignored brightness change (Lights are OFF)")
            return
            
        new_level = max(0, min(100, int(level)))
        if self.brightness != new_level:
            self.brightness = new_level
            # Reduced logging to prevent spamming
            # print(f"[LIGHT] Brightness set to {self.brightness}%")

    def increase_brightness(self, amount=5):
        self.set_brightness(self.brightness + amount)

    def decrease_brightness(self, amount=5):
        self.set_brightness(self.brightness - amount)

    def set_mode(self, mode):
        now = time.time()
        if now - self.last_mode_change < self.mode_debounce_time:
            return  # Ignore rapid mode changes
            
        if not self.is_on:
            self.turn_on()
            
        if self.mode != mode:
            self.mode = mode
            self.last_mode_change = now
            print(f"[LIGHT] Mode switched to: {self.mode}")
            
    def get_status(self):
        return {
            "on": self.is_on,
            "brightness": self.brightness,
            "mode": self.mode
        }
