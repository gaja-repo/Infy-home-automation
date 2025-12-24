import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import time
from utils import draw_text, COLOR_ON, COLOR_OFF

class GestureRecognizer:
    def __init__(self):
        # Get the model path relative to this file
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        
        # Create HandLandmarker options for VIDEO mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,  # Single hand for better stability
            min_hand_detection_confidence=0.7,  # Higher confidence
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        
        # Fixed calibration range for brightness (no auto-calibration)
        self.min_distance = 30   # Fingers touching
        self.max_distance = 220  # Fingers fully spread
        
        # Smoothing for stable brightness
        self.brightness_ema = 50.0  # Exponential moving average
        self.ema_alpha = 0.25       # Smoothing factor (lower = more stable)
        self.last_brightness = 50   # Hold brightness when hand leaves
        
        # Gesture confirmation system
        self.current_gesture = None
        self.gesture_start_time = 0
        self.gesture_hold_time = 0.4  # Seconds to confirm a gesture
        self.confirmed_gesture = None
        
        # Debouncing for on/off actions
        self.last_on_off_time = 0
        self.on_off_cooldown = 1.0  # 1 second cooldown between on/off
        
        # Mode tracking
        self.brightness_mode_active = False
        self.frames_without_hand = 0
        self.hand_lost_threshold = 15  # Frames before considering hand "lost"

    def process_frame(self, frame, light_controller):
        """Process a video frame and detect hand gestures."""
        self.frame_timestamp_ms += 33  # Approximately 30 FPS
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        frame_action = None
        gesture_detected = None
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            self.frames_without_hand = 0
            hand_landmarks = result.hand_landmarks[0]  # Use first hand only
            
            # Draw landmarks on frame
            self._draw_landmarks(frame, hand_landmarks)
            
            # Convert normalized landmarks to pixel coordinates
            h, w, c = frame.shape
            lm_list = []
            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([idx, cx, cy])
            
            if len(lm_list) != 0:
                # Detect finger states
                thumb_open = self._is_thumb_open(lm_list)
                index_open = lm_list[8][2] < lm_list[6][2]
                middle_open = lm_list[12][2] < lm_list[10][2]
                ring_open = lm_list[16][2] < lm_list[14][2]
                pinky_open = lm_list[20][2] < lm_list[18][2]
                
                fingers_open = [thumb_open, index_open, middle_open, ring_open, pinky_open]
                count = sum(fingers_open)
                
                # Gesture Detection with Clear Separation
                now = time.time()
                
                # BRIGHTNESS MODE: Only thumb and index extended (pinch gesture)
                if thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
                    gesture_detected = 'brightness'
                    
                    # Calculate distance for brightness
                    x1, y1 = lm_list[4][1], lm_list[4][2]
                    x2, y2 = lm_list[8][1], lm_list[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Draw visual feedback
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                    
                    # Map distance to brightness with fixed range
                    raw_brightness = ((distance - self.min_distance) / 
                                     (self.max_distance - self.min_distance)) * 100
                    raw_brightness = max(0, min(100, raw_brightness))
                    
                    # Apply EMA smoothing
                    self.brightness_ema = (self.ema_alpha * raw_brightness + 
                                          (1 - self.ema_alpha) * self.brightness_ema)
                    brightness = int(round(self.brightness_ema))
                    brightness = max(0, min(100, brightness))
                    
                    # Store and set brightness
                    self.last_brightness = brightness
                    light_controller.set_brightness(brightness)
                    self.brightness_mode_active = True
                    frame_action = f"Brightness: {brightness}%"
                    
                    # Show distance info
                    cv2.putText(frame, f"Dist: {int(distance)}", (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # ON GESTURE: Open palm (all 5 fingers) or just index finger
                elif count >= 4 or (count == 1 and index_open):
                    gesture_detected = 'on'
                    self.brightness_mode_active = False
                
                # OFF GESTURE: Closed fist (0-1 fingers) or peace sign
                elif count <= 1 and not index_open:
                    gesture_detected = 'off'
                    self.brightness_mode_active = False
                elif count == 2 and index_open and middle_open:
                    gesture_detected = 'off'
                    self.brightness_mode_active = False
                else:
                    gesture_detected = None
                    self.brightness_mode_active = False
                
                # Gesture confirmation with hold time
                if gesture_detected and gesture_detected != 'brightness':
                    if gesture_detected == self.current_gesture:
                        elapsed = now - self.gesture_start_time
                        # Show progress bar
                        progress = min(1.0, elapsed / self.gesture_hold_time)
                        bar_width = int(150 * progress)
                        cv2.rectangle(frame, (10, 65), (10 + bar_width, 80), (0, 255, 255), -1)
                        cv2.rectangle(frame, (10, 65), (160, 80), (100, 100, 100), 2)
                        
                        if elapsed >= self.gesture_hold_time:
                            # Gesture confirmed!
                            if now - self.last_on_off_time > self.on_off_cooldown:
                                if gesture_detected == 'on':
                                    light_controller.turn_on()
                                    frame_action = "Lights ON"
                                    self.last_on_off_time = now
                                elif gesture_detected == 'off':
                                    light_controller.turn_off()
                                    frame_action = "Lights OFF"
                                    self.last_on_off_time = now
                                self.current_gesture = None
                    else:
                        self.current_gesture = gesture_detected
                        self.gesture_start_time = now
        else:
            # No hand detected
            self.frames_without_hand += 1
            self.current_gesture = None
            self.brightness_mode_active = False
            
            # Keep last brightness (don't reset to 0)
            if self.frames_without_hand < self.hand_lost_threshold:
                # Still show last brightness briefly
                pass
        
        # Show gesture mode indicator
        if self.brightness_mode_active:
            cv2.putText(frame, "MODE: Brightness", (450, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        elif self.current_gesture == 'on':
            cv2.putText(frame, "Hold for ON...", (450, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif self.current_gesture == 'off':
            cv2.putText(frame, "Hold for OFF...", (450, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if frame_action:
            color = COLOR_ON if "ON" in frame_action else COLOR_OFF
            if "Brightness" in frame_action:
                color = (255, 0, 255)
            draw_text(frame, frame_action, (10, 50), color)
        
        return frame
    
    def _is_thumb_open(self, lm_list):
        """Check if thumb is open (accounts for left/right hand orientation)."""
        # Thumb is open if tip (4) is further from palm center than knuckle (2)
        # Using horizontal distance for thumb
        thumb_tip_x = lm_list[4][1]
        thumb_ip_x = lm_list[3][1]
        wrist_x = lm_list[0][1]
        
        # If wrist is left of thumb IP, it's right hand (thumb goes right when open)
        if wrist_x < thumb_ip_x:
            return thumb_tip_x > thumb_ip_x
        else:
            return thumb_tip_x < thumb_ip_x
    
    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on the frame."""
        h, w, c = frame.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17)  # Palm
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    
    def close(self):
        """Clean up resources."""
        self.landmarker.close()
