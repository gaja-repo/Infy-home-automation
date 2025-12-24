import cv2
import numpy as np
import pickle
import os

class FaceRecognitionSystem:
    """Face detection and recognition system for authorized user access."""
    
    def __init__(self, max_faces=2, data_file='faces.pkl'):
        self.max_faces = max_faces
        self.data_file = data_file
        self.face_data = {}  # {name: face_encoding}
        
        # Load face cascade for detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Face recognition using LBPH (Local Binary Patterns Histograms)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer_trained = False
        
        # Load existing faces
        self.load_faces()
        
    def register_face(self, name, frame):
        """
        Register a new face with the given name.
        
        Args:
            name: Name of the person
            frame: BGR image frame containing the face
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if len(self.face_data) >= self.max_faces:
            return False, f"Maximum {self.max_faces} faces already registered"
        
        if name in self.face_data:
            return False, f"Face with name '{name}' already exists"
        
        # Detect face in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, "No face detected in image"
        
        if len(faces) > 1:
            return False, "Multiple faces detected, please ensure only one person is visible"
        
        # Extract face region
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))  # Normalize size
        
        # Store face data
        self.face_data[name] = face_roi
        
        # Retrain recognizer
        self._train_recognizer()
        
        # Save to disk
        self.save_faces()
        
        return True, f"Face registered successfully as '{name}'"
    
    def delete_face(self, name):
        """
        Delete a registered face.
        
        Args:
            name: Name of the face to delete
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if name not in self.face_data:
            return False, f"No face found with name '{name}'"
        
        del self.face_data[name]
        
        # Retrain recognizer
        if len(self.face_data) > 0:
            self._train_recognizer()
        else:
            self.recognizer_trained = False
        
        # Save to disk
        self.save_faces()
        
        return True, f"Face '{name}' deleted successfully"
    
    def is_authorized_face(self, frame):
        """
        Check if the frame contains an authorized face.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (authorized: bool, name: str or None)
        """
        if not self.recognizer_trained or len(self.face_data) == 0:
            return True, None  # No faces registered, allow all
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, None  # No face detected
        
        # Check the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Recognize face
        label, confidence = self.face_recognizer.predict(face_roi)
        
        # Lower confidence = better match (it's actually a distance measure)
        # Typical threshold: 50-80 for LBPH
        if confidence < 70:
            name = list(self.face_data.keys())[label]
            return True, name
        
        return False, None
    
    def detect_and_draw_faces(self, frame, authorized_only=False):
        """
        Detect faces in frame and draw rectangles.
        
        Args:
            frame: BGR image frame
            authorized_only: If True, only draw authorized faces
            
        Returns:
            Modified frame with face rectangles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            if authorized_only and self.recognizer_trained:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.face_recognizer.predict(face_roi)
                
                if confidence < 70:
                    name = list(self.face_data.keys())[label]
                    color = (0, 255, 0)  # Green for authorized
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, color, 2)
                else:
                    color = (0, 0, 255)  # Red for unauthorized
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, "Unauthorized", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Just draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        return frame
    
    def get_registered_faces(self):
        """Get list of registered face names."""
        return list(self.face_data.keys())
    
    def get_face_count(self):
        """Get number of registered faces."""
        return len(self.face_data)
    
    def _train_recognizer(self):
        """Train the face recognizer with current face data."""
        if len(self.face_data) == 0:
            self.recognizer_trained = False
            return
        
        faces = []
        labels = []
        
        for idx, (name, face_roi) in enumerate(self.face_data.items()):
            faces.append(face_roi)
            labels.append(idx)
        
        self.face_recognizer.train(faces, np.array(labels))
        self.recognizer_trained = True
    
    def save_faces(self):
        """Save face data to disk."""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.face_data, f)
            print(f"[FACE] Saved {len(self.face_data)} faces to {self.data_file}")
        except Exception as e:
            print(f"[FACE] Error saving faces: {e}")
    
    def load_faces(self):
        """Load face data from disk."""
        if not os.path.exists(self.data_file):
            print("[FACE] No existing face data found")
            return
        
        try:
            with open(self.data_file, 'rb') as f:
                self.face_data = pickle.load(f)
            print(f"[FACE] Loaded {len(self.face_data)} faces from {self.data_file}")
            
            # Retrain recognizer with loaded data
            if len(self.face_data) > 0:
                self._train_recognizer()
        except Exception as e:
            print(f"[FACE] Error loading faces: {e}")
            self.face_data = {}
