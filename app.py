"""
Face Emotion Detection - Flask Web Application
Assignment 2: Bioinformatics Masters Program

This Flask app allows students to:
1. Fill in their information (name, email, student ID)
2. Upload a photo of themselves
3. Get emotion detection results
4. Store data in SQLite database
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import sqlite3
from datetime import datetime
import numpy as np
from tensorflow import keras
from PIL import Image
import cv2
from werkzeug.utils import secure_filename

# ============================================
# FLASK APP CONFIGURATION
# ============================================

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_12345'  # For flash messages

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'face_emotionModel.h5'
DATABASE = 'database.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion messages (personalized responses)
EMOTION_MESSAGES = {
    'Happy': "You are smiling! You look happy! 😊",
    'Sad': "You are frowning. Why are you sad? 😢",
    'Angry': "You look angry! What's bothering you? 😠",
    'Surprise': "You look surprised! Did something unexpected happen? 😮",
    'Fear': "You look fearful. Is everything okay? 😰",
    'Disgust': "You look disgusted. What's wrong? 🤢",
    'Neutral': "You have a neutral expression. Feeling calm? 😐"
}

# ============================================
# LOAD THE TRAINED MODEL
# ============================================

print("Loading trained emotion detection model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# ============================================
# DATABASE SETUP
# ============================================

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            student_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detected_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized!")

# Initialize database on startup
init_db()

# ============================================
# HELPER FUNCTIONS
# ============================================

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect face (optional - if no face detected, use whole image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        # If face detected, crop to face region
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Use first detected face
            img = img[y:y+h, x:x+w]
        
        # Resize to 48x48 (model input size)
        img = cv2.resize(img, (48, 48))
        
        # Normalize pixel values
        img = img / 255.0
        
        # Reshape for model input
        img = img.reshape(1, 48, 48, 1)
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """Predict emotion from image"""
    if model is None:
        return "Error", 0.0, "Model not loaded"
    
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            return "Error", 0.0, "Could not process image"
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get predicted emotion and confidence
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, confidence, EMOTION_MESSAGES[emotion]
    
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return "Error", 0.0, f"Prediction failed: {str(e)}"

def save_to_database(name, email, student_id, image_path, emotion, confidence):
    """Save student information and results to database"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO students (name, email, student_id, image_path, detected_emotion, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, email, student_id, image_path, emotion, confidence))
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        print(f"Database error: {e}")
        return False

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle form submission and image upload"""
    
    # Get form data
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    student_id = request.form.get('student_id', '').strip()
    
    # Validate form data
    if not name or not email or not student_id:
        flash('Please fill in all fields!', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'photo' not in request.files:
        flash('No file uploaded!', 'error')
        return redirect(url_for('index'))
    
    file = request.files['photo']
    
    # Check if file has a name
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type! Please upload PNG, JPG, or JPEG.', 'error')
        return redirect(url_for('index'))
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Predict emotion
    emotion, confidence, message = predict_emotion(filepath)
    
    # Save to database
    save_success = save_to_database(name, email, student_id, filepath, emotion, confidence)
    
    if not save_success:
        flash('Error saving to database!', 'error')
    
    # Display result
    return render_template('result.html',
                         name=name,
                         emotion=emotion,
                         confidence=round(confidence * 100, 2),
                         message=message,
                         image_path=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================
# RUN THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FACE EMOTION DETECTION WEB APP")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)