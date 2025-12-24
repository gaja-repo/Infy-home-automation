import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
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
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        # For distance-based brightness control
        self.min_distance = 20  # Will auto-calibrate
        self.max_distance = 200  # Will auto-calibrate
        self.calibration_samples = []

    def process_frame(self, frame, light_controller):
        """Process a video frame and detect hand gestures."""
        self.frame_timestamp_ms += 33  # Approximately 30 FPS
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        frame_action = None
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw landmarks on frame
                self._draw_landmarks(frame, hand_landmarks)
                
                # Convert normalized landmarks to pixel coordinates
                h, w, c = frame.shape
                lm_list = []
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([idx, cx, cy])
                
                if len(lm_list) != 0:
                    # Finger counting logic
                    # Tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
                    # PIP: 6, 10, 14, 18
                    
                    # Check which fingers are open (tip above PIP in Y axis)
                    index_open = lm_list[8][2] < lm_list[6][2]
                    middle_open = lm_list[12][2] < lm_list[10][2]
                    ring_open = lm_list[16][2] < lm_list[14][2]
                    pinky_open = lm_list[20][2] < lm_list[18][2]
                    
                    count = sum([index_open, middle_open, ring_open, pinky_open])
                    
                    # 1 Finger (Index only) = Lights ON
                    if count == 1 and index_open:
                        light_controller.turn_on()
                        frame_action = "Lights ON"
                    # 2 Fingers (Index + Middle) = Lights OFF
                    elif count == 2 and index_open and middle_open:
                        light_controller.turn_off()
                        frame_action = "Lights OFF"
                    
                    # Distance-based Brightness Control (Thumb tip 4 and Index tip 8)
                    x1, y1 = lm_list[4][1], lm_list[4][2]
                    x2, y2 = lm_list[8][1], lm_list[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Only control brightness with thumb and index (other fingers closed)
                    if not middle_open and not ring_open and not pinky_open:
                        # Draw distance line
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.circle(frame, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                        
                        # Auto-calibrate min/max distances for better accuracy
                        self.calibration_samples.append(distance)
                        if len(self.calibration_samples) > 30:  # Keep last 30 samples
                            self.calibration_samples.pop(0)
                            self.min_distance = min(self.calibration_samples) * 0.9
                            self.max_distance = max(self.calibration_samples) * 1.1
                        
                        # Map distance to brightness (0-100%)
                        # Fully closed (min distance) = 0%, Fully open (max distance) = 100%
                        brightness = int(((distance - self.min_distance) / 
                                        (self.max_distance - self.min_distance)) * 100)
                        brightness = max(0, min(100, brightness))  # Clamp to 0-100
                        
                        # Set brightness directly
                        light_controller.set_brightness(brightness)
                        frame_action = f"Brightness: {brightness}%"
                        
                        # Visual feedback: show distance gauge
                        cv2.putText(frame, f"Dist: {int(distance)}", (cx + 10, cy - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        if frame_action:
            color = COLOR_ON if "ON" in frame_action or "++" in frame_action else COLOR_OFF
            draw_text(frame, frame_action, (10, 50), color)
        
        return frame
    
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
