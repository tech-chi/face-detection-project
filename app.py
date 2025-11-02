import os
import sqlite3
import numpy as np
import cv2  # OpenCV for image processing
import tensorflow as tf
from flask import Flask, request, render_template, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import click

# --- Configuration ---
app = Flask(__name__)
# Render provides the DATABASE_URL env var
# Format: postgresql://user:password@host:port/database
db_url = os.environ.get('DATABASE_URL')

if not db_url:
    print("Error: DATABASE_URL environment variable not set.")
    # We'll set a local fallback for 'flask init-db' to work,
    # but the app will fail on Render if this isn't set.
    # We will create a local 'instance' folder for this fallback.
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    db_url = f"sqlite:///{os.path.join(app.instance_path, 'local.db')}"
    print(f"Warning: Using local fallback SQLite DB at {db_url}")

app.config['SQLALCHEMY_DATABASE_URI'] = db_url.replace("postgres://", "postgresql://", 1) # Fix for SQLAlchemy
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

db = SQLAlchemy(app)

MODEL_FILE = 'face_emotionModel.h5'
IMG_WIDTH, IMG_HEIGHT = 48, 48

# --- Emotion Mapping ---
EMOTION_MAP = {
    0: ("Angry", "You seem angry. What's troubling you?"),
    1: ("Disgust", "You look disgusted. Did something unpleasant happen?"),
    2: ("Fear", "You appear fearful. Is everything alright?"),
    3: ("Happy", "You are smiling! What's the great news?"),
    4: ("Sad", "You are frowning. Why are you sad?"),
    5: ("Surprise", "You look surprised! What happened?"),
    6: ("Neutral", "You seem neutral. Just a regular day?")
}

# --- Load Model ---
print(f"Loading emotion model from {MODEL_FILE}...")
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_FILE}' is in the root directory.")
    model = None

# --- Database Model ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    # Use LargeBinary for BLOB storage (works on Postgres and SQLite)
    image_data = db.Column(db.LargeBinary, nullable=False)
    detected_emotion = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now())

# --- Database Setup Command ---
@app.cli.command('init-db')
def init_db_command():
    """Flask CLI command to initialize the database tables."""
    try:
        with app.app_context():
            db.create_all()
        print('Initialized the database tables.')
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Please ensure your DATABASE_URL is set correctly in Render.")

# --- Image Preprocessing ---
def preprocess_image(image_file_storage):
    try:
        in_memory_file = np.frombuffer(image_file_storage.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Could not decode image.")

        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_array = img_resized / 255.0
        img_array = img_array.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
        
        return img_array, True, None
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None, False, f"Error processing image: {e}"

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    error_text = None

    if request.method == 'POST':
        if model is None:
            error_text = "Model is not loaded. Cannot perform prediction."
            return render_template('index.html', error_text=error_text)

        name = request.form['name']
        email = request.form['email']
        
        if 'image' not in request.files or not request.files['image'].filename:
            error_text = "No image file selected."
            return render_template('index.html', error_text=error_text)

        image_file = request.files['image']
        filename = image_file.filename
        
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            error_text = "Invalid file type. Please upload a .png, .jpg, or .jpeg"
        else:
            image_file.seek(0)
            image_data_for_db = image_file.read() 
            image_file.seek(0) 
            
            processed_image, success, error = preprocess_image(image_file)
            
            if not success:
                error_text = error
            else:
                prediction = model.predict(processed_image)
                pred_index = np.argmax(prediction)
                emotion_label, response_text = EMOTION_MAP.get(pred_index, ("Unknown", "Couldn't read that expression."))
                
                prediction_text = response_text
                
                try:
                    # New save method using SQLAlchemy
                    new_user = User(
                        name=name, 
                        email=email, 
                        image_data=image_data_for_db, 
                        detected_emotion=emotion_label
                    )
                    db.session.add(new_user)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"Database save error: {e}")
                    error_text = "Prediction complete, but failed to save data to DB."

    return render_template('index.html', prediction_text=prediction_text, error_text=error_text)

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', error_text="File is too large. Max size is 5 MB."), 413

