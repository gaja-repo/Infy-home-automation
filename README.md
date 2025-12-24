# 🏠 Enhanced Home Automation System

Gesture & voice controlled smart lighting with face recognition and web dashboard.

## ✨ Features

### 🖐️ Distance-Based Brightness Control
- Control brightness by moving thumb and index finger apart/together
- **Fully closed** = 0% brightness
- **Fully open** = 100% brightness  
- **Auto-calibrating** for smooth, adaptive control

### 👏 Triple Clap Detection
- **Single Clap** → Normal Mode
- **Double Clap** → Relaxing Mode
- **Triple Clap** → Party Mode

### 🔒 Face Recognition Security
- Register up to **2 authorized faces**
- Gestures/claps only work for authorized users
- Real-time visual authorization feedback

### 🌐 Web Dashboard
- Face registration & management
- Live camera feed
- Manual controls for testing
- Real-time status monitoring
- Premium dark theme with glassmorphism

---

## 🚀 Quick Start

### 1. Install Dependencies
```batch
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Dashboard (Optional but Recommended)
```batch
run_dashboard.bat
```
Then open browser to: **http://localhost:5000**

### 3. Register Faces (via Dashboard)
1. Enter your name
2. Click "Capture & Register"
3. Repeat for second user (optional)

### 4. Run Main Application
```batch
run.bat
```

---

## 🎮 Controls

### Hand Gestures
| Gesture | Action |
|---------|--------|
| 👆 **1 Finger** (Index) | Lights ON |
| ✌️ **2 Fingers** (Index + Middle) | Lights OFF |
| 🤏 **Thumb-Index Distance** | Brightness 0-100% |

### Clap Patterns
| Pattern | Mode |
|---------|------|
| 👏 **Single** | Normal |
| 👏👏 **Double** | Relaxing |
| 👏👏👏 **Triple** | Party |

---

## 📁 Files

- `main.py` - Main camera & gesture application
- `dashboard_server.py` - Web dashboard server
- `face_recognition_system.py` - Face recognition module
- `gesture_control.py` - Hand gesture recognition
- `audio_control.py` - Clap detection
- `light_controller.py` - Light control logic

---

## 🔧 Configuration

### Adjust Microphone Sensitivity
Edit `audio_control.py` line 13:
```python
self.threshold = 1000  # Increase if too sensitive, decrease if not detecting
```

### Change Max Faces
Edit initialization in `main.py` or `dashboard_server.py`:
```python
face_system = FaceRecognitionSystem(max_faces=2)  # Change number here
```

---

## 📝 Notes

- Face recognition requires `opencv-contrib-python`
- Gestures disabled when no authorized face detected (if faces registered)
- Dashboard and main app can run simultaneously
- Face data stored in `faces.pkl`

---

## 🆘 Troubleshooting

**Gestures not working?**
- Check if your face is detected (green box = authorized)
- Ensure correct finger positions

**Claps not detected?**
- Adjust microphone threshold
- Try clapping louder

**Can't register face?**
- Ensure good lighting
- Only one person visible to camera

---

## 🎓 For More Info

See [`walkthrough.md`](file:///C:/Users/ADMIN/.gemini/antigravity/brain/55a3d809-5652-481b-8788-5e02fd6f1400/walkthrough.md) for complete technical documentation.
