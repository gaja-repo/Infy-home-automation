import cv2
from light_controller import LightController
from gesture_control import GestureRecognizer
from audio_control import AudioController
from face_recognition_system import FaceRecognitionSystem
from utils import WEBCAM_ID, WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT, draw_text, COLOR_OFF, COLOR_ON

def main():
    # Initialize Controllers
    light_controller = LightController()
    gesture_recognizer = GestureRecognizer()
    audio_controller = AudioController(light_controller)
    face_system = FaceRecognitionSystem(max_faces=2)

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

    print("="*60)
    print("Home Automation System Started")
    print("="*60)
    if face_system.get_face_count() > 0:
        print(f"Face Recognition: ENABLED ({face_system.get_face_count()} faces registered)")
        print("Registered faces:", ", ".join(face_system.get_registered_faces()))
    else:
        print("Face Recognition: DISABLED (No faces registered)")
        print("Visit http://localhost:5000 to register faces via dashboard")
    print("Press 'q' to exit")
    print("="*60)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            # Check for authorized face
            is_authorized, detected_name = face_system.is_authorized_face(frame)
            
            # Draw face detection rectangles
            frame = face_system.detect_and_draw_faces(frame, authorized_only=True)
            
            if is_authorized:
                # Process Gestures only if authorized
                frame = gesture_recognizer.process_frame(frame, light_controller)
                
                # Show authorized status
                if detected_name:
                    cv2.rectangle(frame, (0, 420), (200, 480), (0, 255, 0), cv2.FILLED)
                    draw_text(frame, f"Authorized: {detected_name}", (10, 450), COLOR_ON, 0.5)
            else:
                # Show unauthorized message
                if face_system.get_face_count() > 0:
                    cv2.rectangle(frame, (0, 420), (320, 480), (0, 0, 255), cv2.FILLED)
                    draw_text(frame, "UNAUTHORIZED - Show Face", (10, 450), COLOR_OFF, 0.6)

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
        gesture_recognizer.close()
        audio_controller.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("System Shut Down.")

if __name__ == "__main__":
    main()
