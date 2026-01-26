# Face Emotion Detection System

### Description
AI-powered web application that detects emotions from facial images using deep learning.

### Features
- Upload photo interface
- Real-time emotion detection
- 7 emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- SQLite database storage
- Beautiful responsive UI

### Technologies
- Python, TensorFlow/Keras
- Flask web framework
- OpenCV for image processing
- CNN deep learning model
- SQLite database

### Model Performance
- Training Accuracy: 58.07%
- Model: Custom CNN architecture
- Dataset: FER2013 (35,887 images)

### Local Setup
```
pip install -r requirements.txt
python model_training.py  # Train model
python app.py             # Run web app
```

### Deployment
Deployed on Render: https://face-emotion-detection-wl8x.onrender.com

### Author
**Student:** Jimoh-Alabi Islamiat Modupeoluwa  
**Program:** Masters in Bioinformatics
