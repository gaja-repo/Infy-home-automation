import cv2

# Constants
WEBCAM_ID = 0
WINDOW_NAME = "Home Automation Controller"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Colors (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_ON = (0, 255, 0)
COLOR_OFF = (0, 0, 255)

def draw_text(image, text, position, color=COLOR_TEXT, scale=0.7):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
