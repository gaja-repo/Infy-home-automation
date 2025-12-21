class LightController:
    """
    Simulates a home automation light controller.
    In a real system, this would interface with hardware (Hue, Tuya, GPIO, etc.).
    """
    def __init__(self):
        self.is_on = False
        self.brightness = 50  # 0 to 100
        self.mode = "Normal"  # Normal, Relaxing, Party

    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            print("[LIGHT] Turned ON")

    def turn_off(self):
        if self.is_on:
            self.is_on = False
            print("[LIGHT] Turned OFF")

    def set_brightness(self, level):
        """Sets brightness between 0 and 100."""
        if not self.is_on:
            print("[LIGHT] Ignored brightness change (Lights are OFF)")
            return
            
        new_level = max(0, min(100, int(level)))
        if self.brightness != new_level:
            self.brightness = new_level
            print(f"[LIGHT] Brightness set to {self.brightness}%")

    def increase_brightness(self, amount=5):
        self.set_brightness(self.brightness + amount)

    def decrease_brightness(self, amount=5):
        self.set_brightness(self.brightness - amount)

    def set_mode(self, mode):
        if not self.is_on:
            self.turn_on()
            
        if self.mode != mode:
            self.mode = mode
            print(f"[LIGHT] Mode switched to: {self.mode}")
            
    def get_status(self):
        return {
            "on": self.is_on,
            "brightness": self.brightness,
            "mode": self.mode
        }
