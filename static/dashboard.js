// Dashboard State
let currentStatus = {};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function () {
    console.log('Dashboard initialized');
    updateStatus();
    setInterval(updateStatus, 2000); // Update every 2 seconds

    // Event listeners
    document.getElementById('capture-btn').addEventListener('click', registerFace);
    document.getElementById('toggle-light-btn').addEventListener('click', toggleLight);
    document.getElementById('brightness-slider').addEventListener('input', updateBrightnessDisplay);
    document.getElementById('brightness-slider').addEventListener('change', setBrightness);

    // Mode buttons
    document.querySelectorAll('.btn-mode').forEach(btn => {
        btn.addEventListener('click', function () {
            testMode(this.dataset.mode);
        });
    });
});

// Update system status
async function updateStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();
        currentStatus = status;

        // Update UI
        document.getElementById('light-status').textContent = status.on ? 'ON' : 'OFF';
        document.getElementById('light-status').style.color = status.on ? '#00e676' : '#ff5252';

        document.getElementById('brightness-status').textContent = status.brightness + '%';
        document.getElementById('mode-status').textContent = status.mode;
        document.getElementById('face-count').textContent = `${status.face_count} / ${status.max_faces}`;

        // Update mode badge color
        const modeBadge = document.getElementById('mode-status');
        modeBadge.style.background = getModeColor(status.mode);

        // Update registered faces list
        updateFacesList(status.registered_faces);

        // Highlight active mode
        document.querySelectorAll('.btn-mode').forEach(btn => {
            if (btn.dataset.mode === status.mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Update faces list
function updateFacesList(faces) {
    const facesList = document.getElementById('faces-list');

    if (faces.length === 0) {
        facesList.innerHTML = '<p class="empty-message">No faces registered yet</p>';
        return;
    }

    facesList.innerHTML = faces.map(name => `
        <div class="face-item">
            <span class="face-name">👤 ${name}</span>
            <button class="btn btn-danger" onclick="deleteFace('${name}')">Delete</button>
        </div>
    `).join('');
}

// Register new face
async function registerFace() {
    const nameInput = document.getElementById('face-name');
    const name = nameInput.value.trim();

    if (!name) {
        showNotification('Please enter a name', 'error');
        return;
    }

    try {
        const response = await fetch('/register_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });

        const result = await response.json();

        if (result.success) {
            showNotification(result.message, 'success');
            nameInput.value = '';
            updateStatus();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Error registering face: ' + error.message, 'error');
    }
}

// Delete face
async function deleteFace(name) {
    if (!confirm(`Are you sure you want to delete "${name}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/delete_face/${name}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (result.success) {
            showNotification(result.message, 'success');
            updateStatus();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Error deleting face: ' + error.message, 'error');
    }
}

// Toggle light
async function toggleLight() {
    try {
        const response = await fetch('/toggle_light', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            showNotification(result.message, 'success');
            updateStatus();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Error toggling light: ' + error.message, 'error');
    }
}

// Update brightness display
function updateBrightnessDisplay() {
    const slider = document.getElementById('brightness-slider');
    const valueDisplay = document.getElementById('brightness-value');
    valueDisplay.textContent = slider.value + '%';
}

// Set brightness
async function setBrightness() {
    const brightness = document.getElementById('brightness-slider').value;

    try {
        const response = await fetch('/set_brightness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ brightness: parseInt(brightness) })
        });

        const result = await response.json();

        if (result.success) {
            showNotification(result.message, 'success');
            updateStatus();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Error setting brightness: ' + error.message, 'error');
    }
}

// Test mode
async function testMode(mode) {
    try {
        const response = await fetch('/test_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode })
        });

        const result = await response.json();

        if (result.success) {
            showNotification(result.message, 'success');
            updateStatus();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Error setting mode: ' + error.message, 'error');
    }
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type} show`;

    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Get mode color
function getModeColor(mode) {
    const colors = {
        'Normal': '#6c5ce7',
        'Relaxing': '#00b8d4',
        'Party': '#ff5252'
    };
    return colors[mode] || '#6c5ce7';
}
