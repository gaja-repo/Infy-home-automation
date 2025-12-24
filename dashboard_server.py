from flask import Flask, render_template, request, jsonify, Response
import cv2
import json
import threading
import time
from light_controller import LightController
from face_recognition_system import FaceRecognitionSystem

app = Flask(__name__)

# Shared state
light_controller = LightController()
face_system = FaceRecognitionSystem(max_faces=2)
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Get or initialize camera."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(3, 640)
        camera.set(4, 480)
    return camera

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/status')
def get_status():
    """Get current system status."""
    status = light_controller.get_status()
    status['registered_faces'] = face_system.get_registered_faces()
    status['face_count'] = face_system.get_face_count()
    status['max_faces'] = face_system.max_faces
    return jsonify(status)

@app.route('/register_face', methods=['POST'])
def register_face():
    """Register a new face."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'}), 400
        
        # Capture frame from camera
        with camera_lock:
            cam = get_camera()
            ret, frame = cam.read()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture image'}), 500
        
        # Register face
        success, message = face_system.register_face(name, frame)
        
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete_face/<name>', methods=['DELETE'])
def delete_face(name):
    """Delete a registered face."""
    try:
        success, message = face_system.delete_face(name)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/test_mode', methods=['POST'])
def test_mode():
    """Manually trigger a mode for testing."""
    try:
        data = request.get_json()
        mode = data.get('mode', 'Normal')
        
        if mode not in ['Normal', 'Relaxing', 'Party']:
            return jsonify({'success': False, 'message': 'Invalid mode'}), 400
        
        light_controller.set_mode(mode)
        return jsonify({'success': True, 'message': f'Mode set to {mode}'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/set_brightness', methods=['POST'])
def set_brightness():
    """Set brightness level."""
    try:
        data = request.get_json()
        brightness = int(data.get('brightness', 50))
        
        light_controller.set_brightness(brightness)
        return jsonify({'success': True, 'message': f'Brightness set to {brightness}%'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/toggle_light', methods=['POST'])
def toggle_light():
    """Toggle light on/off."""
    try:
        status = light_controller.get_status()
        if status['on']:
            light_controller.turn_off()
            message = 'Light turned OFF'
        else:
            light_controller.turn_on()
            message = 'Light turned ON'
        
        return jsonify({'success': True, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/capture_preview')
def capture_preview():
    """Capture a single frame for preview."""
    try:
        with camera_lock:
            cam = get_camera()
            ret, frame = cam.read()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture'}), 500
        
        # Draw face detection rectangles
        frame = face_system.detect_and_draw_faces(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to encode image'}), 500
        
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def generate_frames():
    """Generate frames for video streaming."""
    cam = get_camera()
    while True:
        with camera_lock:
            success, frame = cam.read()
        
        if not success:
            break
        
        # Draw face detection
        frame = face_system.detect_and_draw_faces(frame, authorized_only=True)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 fps

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*50)
    print("Home Automation Dashboard")
    print("="*50)
    print("Dashboard running at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
