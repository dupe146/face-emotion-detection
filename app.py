""""
Face Emotion Detection - Flask Web Application
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import sqlite3
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
import base64

# ============================================
# FLASK APP CONFIGURATION
# ============================================

# Get the absolute path to the project directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask with explicit paths
app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'))

app.secret_key = 'your_secret_key_here_12345'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'face_emotionModel.h5'
DATABASE = 'database.db'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create uploads folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
# DOWNLOAD MODEL IF NOT EXISTS
# ============================================
def download_model():
    """Download model from Google Drive if not present"""
    if os.path.exists(MODEL_PATH):
        print("✓ Model file found locally!")
        return True
        
    print("Model not found. Downloading from Google Drive...")
    try:
        import gdown
        
        # Replace YOUR_FILE_ID with the actual ID from Google Drive link
        file_id = "1m3QdxrBX0PvEGC-FtPVEhSvIukafZos8"  # ← PASTE YOUR FILE_ID HERE!
        url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"Downloading from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        
        # Verify download
        if os.path.exists(MODEL_PATH):
            size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✓ Model downloaded successfully! Size: {size:.2f} MB")
            return True
        else:
            print("✗ Download verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("Please check:")
        print("1. Google Drive link is public")
        print("2. File ID is correct")
        print("3. gdown package is installed")
        return False

# ============================================
# LOAD THE TRAINED MODEL (VERSION-SAFE)
# ============================================

model = None

def load_model_safe():
    """Load model with TensorFlow version compatibility"""
    global model
    
    try:
        # First, ensure model file exists
        if not download_model():
            print("ERROR: Could not get model file!")
            return False
        
        print(f"Loading model from: {MODEL_PATH}")
        
        # Import TensorFlow
        from tensorflow import keras
        import tensorflow as tf
        
        # Try loading with safe_mode=False for old models
        try:
            model = keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                safe_mode=False
            )
            print("✓ Model loaded successfully!")
            
        except Exception as load_error:
            print(f"Standard load failed: {load_error}")
            print("Trying alternative method...")
            
            # Try with custom_objects
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'InputLayer': tf.keras.layers.InputLayer},
                compile=False
            )
            print("✓ Model loaded with custom_objects!")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model ready for predictions!")
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize model on startup
print("\n" + "="*60)
print("INITIALIZING EMOTION DETECTION MODEL")
print("="*60)

model_loaded = load_model_safe()

if not model_loaded:
    print("⚠️  WARNING: App running without model!")
    model = None
else:
    print("✓ Ready for emotion detection!")

print("="*60 + "\n")
# ============================================
# DATABASE SETUP
# ============================================

def init_db():
    """Initialize database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
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

init_db()

# ============================================
# HELPER FUNCTIONS
# ============================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_webcam_image(base64_data):
    """Save base64 webcam image to file"""
    try:
        # Remove data URL prefix
        image_data = base64_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"webcam_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return filepath, filename
    except Exception as e:
        print(f"Error saving webcam image: {e}")
        return None, None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        # Crop to face if detected
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            img = img[y:y+h, x:x+w]
        
        # Resize to 48x48
        img = cv2.resize(img, (48, 48))
        
        # Normalize
        img = img / 255.0
        
        # Reshape
        img = img.reshape(1, 48, 48, 1)
        
        return img
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def predict_emotion(image_path):
    """Predict emotion from image"""
    if model is None:
        return "Error", 0.0, "Model not loaded"
    
    try:
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            return "Error", 0.0, "Could not process image"
        
        # Predict
        predictions = model.predict(processed_img, verbose=0)
        
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, confidence, EMOTION_MESSAGES[emotion]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, f"Prediction failed: {str(e)}"

def save_to_database(name, email, student_id, image_path, emotion, confidence):
    """Save to database"""
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
# ROUTES
# ============================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and prediction"""
    
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        student_id = request.form.get('student_id', '').strip()
        
        # Validate
        if not name or not email or not student_id:
            flash('Please fill all fields!', 'error')
            return redirect(url_for('index'))
        
        # Ensure uploads folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Check for webcam data or file upload
        webcam_data = request.form.get('webcam_data', '')
        
        if webcam_data:
            # Handle webcam
            print("Processing webcam image...")
            filepath, filename = save_webcam_image(webcam_data)
            if not filepath:
                flash('Error saving webcam image!', 'error')
                return redirect(url_for('index'))
        else:
            # Handle file upload
            if 'photo' not in request.files:
                flash('No file uploaded!', 'error')
                return redirect(url_for('index'))
            
            file = request.files['photo']
            
            if file.filename == '':
                flash('No file selected!', 'error')
                return redirect(url_for('index'))
            
            if not allowed_file(file.filename):
                flash('Invalid file type!', 'error')
                return redirect(url_for('index'))
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        print(f"Image saved: {filepath}")
        
        # Check model
        if model is None:
            flash('Model not available!', 'error')
            return redirect(url_for('index'))
        
        # Predict (with timeout protection)
        print("Predicting emotion...")
        emotion, confidence, message = predict_emotion(filepath)
        
        if emotion == "Error":
            flash(f'Prediction error: {message}', 'error')
            return redirect(url_for('index'))
        
        print(f"Result: {emotion} ({confidence*100:.1f}%)")
        
        # Save to database
        save_to_database(name, email, student_id, filepath, emotion, confidence)
        
        # Show result
        return render_template('result.html',
                             name=name,
                             emotion=emotion,
                             confidence=round(confidence * 100, 2),
                             message=message,
                             image_path=filename)
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FACE EMOTION DETECTION APP")
    print("="*60)
    print(f"Model loaded: {'Yes' if model else 'No'}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)