import cv2
from light_controller import LightController
from gesture_control import GestureRecognizer
from audio_control import AudioController
from utils import WEBCAM_ID, WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT, draw_text

def main():
    # Initialize Controllers
    light_controller = LightController()
    gesture_recognizer = GestureRecognizer()
    audio_controller = AudioController(light_controller)

    # Start Audio Thread
    try:
        audio_controller.start()
    except Exception as e:
        print(f"Error starting audio controller: {e}")
        print("Continuing without audio control...")

    # Start Video Capture
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    print("System Started. Press 'q' to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            # Process Gestures
            frame = gesture_recognizer.process_frame(frame, light_controller)

            # Display Status
            status = light_controller.get_status()
            status_text = f"Light: {'ON' if status['on'] else 'OFF'} | Bri: {status['brightness']}% | Mode: {status['mode']}"
            cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), cv2.FILLED)
            draw_text(frame, status_text, (10, 30), scale=0.6)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        audio_controller.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("System Shut Down.")

if __name__ == "__main__":
    main()
