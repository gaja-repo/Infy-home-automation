import cv2
import numpy as np
import pickle
import os

class FaceRecognitionSystem:
    """
    Robust face detection and recognition system.
    Uses Enhanced Haar Cascade with CLAHE for low-light resilience.
    Uses multi-sample augmentation for angle tolerance.
    """
    
    def __init__(self, max_faces=2, data_file='faces.pkl'):
        self.max_faces = max_faces
        self.data_file = data_file
        self.face_data = {}  # {name: [list of face samples]}
        
        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Face recognition using LBPH with optimized parameters
        # Radius=2, Neighbors=16, Grid=8x8 for better detail capture
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=150
        )
        self.recognizer_trained = False
        
        # CLAHE for low-light enhancement
        # ClipLimit=3.0 allows for stronger contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Recognition threshold (Lower = stricter, Higher = looser)
        self.recognition_threshold = 120
        
        # Stability tracking
        self.recognition_history = []
        self.history_size = 5
        self.last_confirmed_name = None
        
        # Load existing faces
        self.load_faces()
    
    def _enhance_image(self, img):
        """Enhance image for better detection in low light."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = self.clahe.apply(gray)
        return enhanced
    
    def _detect_faces(self, frame):
        """Detect faces using enhanced Haar parameters."""
        enhanced = self._enhance_image(frame)
        
        # Optimized parameters for distance and angles:
        # scaleFactor=1.05: Slower but detects faces at more scales (distance)
        # minNeighbors=3: Lower threshold for detection (angles/occlusion)
        # minSize=(20, 20): Detect smaller faces (distance)
        faces = self.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(500, 500)
        )
        return faces
    
    def register_face(self, name, frame):
        """Register a new face with multi-angle augmentation."""
        if len(self.face_data) >= self.max_faces:
            return False, f"Maximum {self.max_faces} faces already registered"
        
        if name in self.face_data:
            return False, f"Face '{name}' already exists"
        
        faces = self._detect_faces(frame)
        
        if len(faces) == 0:
            return False, "No face detected. Ensure good lighting."
        if len(faces) > 1:
            return False, "Multiple faces detected. One person only."
        
        x, y, w, h = faces[0]
        # Extract face from enhanced image
        enhanced = self._enhance_image(frame)
        face_roi = enhanced[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Create augmented samples
        samples = self._create_augmented_samples(face_roi)
        
        self.face_data[name] = samples
        self._train_recognizer()
        self.save_faces()
        
        return True, f"Registered '{name}' with {len(samples)} samples"
    
    def _create_augmented_samples(self, face_base):
        """Generate variations for robust recognition."""
        samples = [face_base]
        
        # 1. Flip (Mirror)
        samples.append(cv2.flip(face_base, 1))
        
        # 2. Brightness simulation (even on enhanced image)
        for beta in [30, -30]:
            varied = cv2.convertScaleAbs(face_base, alpha=1.0, beta=beta)
            samples.append(varied)
            
        # 3. Rotation (simulate head tilt)
        h, w = face_base.shape
        center = (w//2, h//2)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_base, M, (w, h))
            samples.append(rotated)
            
        # 4. Zoom/Scale (simulate distance)
        for scale in [0.9, 1.1]:
            resized = cv2.resize(face_base, (0,0), fx=scale, fy=scale)
            resized = cv2.resize(resized, (200, 200))
            samples.append(resized)
            
        return samples

    def delete_face(self, name):
        if name not in self.face_data:
            return False, "Face not found"
        del self.face_data[name]
        self._train_recognizer()
        self.save_faces()
        return True, "Deleted successfully"
    
    def is_authorized_face(self, frame):
        """Check for authorized face with stability smoothing."""
        if not self.recognizer_trained:
            return True, None
            
        faces = self._detect_faces(frame)
        
        if len(faces) == 0:
            self.recognition_history.append(None)
            if len(self.recognition_history) > self.history_size:
                self.recognition_history.pop(0)
            return False, None
            
        # Largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        enhanced = self._enhance_image(frame)
        face_roi = enhanced[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        label, confidence = self.face_recognizer.predict(face_roi)
        
        recognized_name = None
        if confidence < self.recognition_threshold:
            recognized_name = list(self.face_data.keys())[label]
            
        # Update history
        self.recognition_history.append(recognized_name)
        if len(self.recognition_history) > self.history_size:
            self.recognition_history.pop(0)
            
        # Majority vote
        counts = {}
        for name in self.recognition_history:
            if name:
                counts[name] = counts.get(name, 0) + 1
        
        if counts:
            best_match = max(counts, key=counts.get)
            if counts[best_match] >= 3:
                self.last_confirmed_name = best_match
                return True, best_match
                
        # Stickiness
        if self.last_confirmed_name and counts.get(self.last_confirmed_name, 0) >= 1:
            return True, self.last_confirmed_name
            
        return False, None

    def detect_and_draw_faces(self, frame, authorized_only=False):
        """Draw faces with status."""
        faces = self._detect_faces(frame)
        enhanced = self._enhance_image(frame)
        
        for (x, y, w, h) in faces:
            color = (255, 255, 0)
            name_text = "Unknown"
            
            if authorized_only and self.recognizer_trained:
                face_roi = enhanced[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence < self.recognition_threshold:
                    name = list(self.face_data.keys())[label]
                    color = (0, 255, 0)
                    name_text = f"{name} ({int(100-confidence/2)}%)"
                else:
                    color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
        return frame

    def _train_recognizer(self):
        if not self.face_data:
            self.recognizer_trained = False
            return
            
        faces = []
        labels = []
        for idx, (name, samples) in enumerate(self.face_data.items()):
            for s in samples:
                faces.append(s)
                labels.append(idx)
                
        if faces:
            self.face_recognizer.train(faces, np.array(labels))
            self.recognizer_trained = True
            print(f"[FACE] Trained on {len(faces)} samples")

    def save_faces(self):
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.face_data, f)
        except Exception as e:
            print(f"[FACE] Save error: {e}")

    def load_faces(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    self.face_data = pickle.load(f)
                self._train_recognizer()
            except:
                self.face_data = {}
