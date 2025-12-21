import cv2
import mediapipe as mp
import math
from utils import draw_text, COLOR_ON, COLOR_OFF

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_pinch_dist = None

    def process_frame(self, frame, light_controller):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        frame_action = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Finger counting (Naive approach)
                # Tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
                # PIP: 6, 10, 14, 18
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                if len(lm_list) != 0:
                    fingers = []
                    
                    # Thumb (check x coordinate relative to other fingers depending on hand side... simplifying for demo)
                    # Simplified: Thumb tip is above thumb mcp? No, thumb is side.
                    # Let's count raised fingers (Index, Middle, Ring, Pinky) based on Y
                    
                    # Thumb: 4, 3 (IP), 2 (MCP)
                    # Check if 4 is "extended" relative to 3.
                    # This is tricky without knowing hand side.
                    # Alternative: Distance logic for pinch.
                    
                    # 1. 1 Finger (Index) & 2 Fingers (Index+Middle) Logic
                    # Check Index(8) < PIP(6) (y-axis inverted in image) -> OPEN
                    index_open = lm_list[8][2] < lm_list[6][2]
                    middle_open = lm_list[12][2] < lm_list[10][2]
                    ring_open = lm_list[16][2] < lm_list[14][2]
                    pinky_open = lm_list[20][2] < lm_list[18][2]
                    
                    count = 0
                    if index_open: count += 1
                    if middle_open: count += 1
                    if ring_open: count += 1
                    if pinky_open: count += 1
                    
                    # Ignore thumb for count to avoid confusion with pinch
                    
                    if count == 1 and index_open:
                        light_controller.turn_on()
                        frame_action = "Lights ON"
                    elif count == 2 and index_open and middle_open:
                        light_controller.turn_off()
                        frame_action = "Lights OFF"

                    # 2. Pinch Logic (Thumb 4 and Index 8)
                    x1, y1 = lm_list[4][1], lm_list[4][2]
                    x2, y2 = lm_list[8][1], lm_list[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    # Only pinch if not doing the 1/2 finger gesture (which usually implies open hand or specific fingers)
                    # If other fingers are closed? 
                    # Let's check if Middle/Ring/Pinky are CLOSED to confirm pinch mode
                    if not middle_open and not ring_open and not pinky_open:
                        # Draw pinch line
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                        if self.last_pinch_dist is not None:
                            # Sensitivity threshold
                            if length > self.last_pinch_dist + 5: # Pinch Out
                                light_controller.increase_brightness()
                                frame_action = "Brightness ++"
                            elif length < self.last_pinch_dist - 5: # Pinch In
                                light_controller.decrease_brightness()
                                frame_action = "Brightness --"
                        
                        self.last_pinch_dist = length
                    else:
                        self.last_pinch_dist = None

        if frame_action:
            draw_text(frame, frame_action, (10, 50), COLOR_ON if "ON" in frame_action or "++" in frame_action else COLOR_OFF)
            
        return frame
