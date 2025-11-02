import os
import sqlite3
import numpy as np
import cv2  # OpenCV for image processing
import tensorflow as tf
from flask import Flask, request, render_template, g
import click

# --- Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB
DATABASE = 'database.db'
MODEL_FILE = 'face_emotionModel.h5'
IMG_WIDTH, IMG_HEIGHT = 48, 48

# Ensure the instance folder exists
# This is where the persistent disk will be mounted on Render
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

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

# --- Database Setup ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        # Use the instance path for the database file
        db_path = os.path.join(app.instance_path, DATABASE)
        db = g._database = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database from the schema.sql file."""
    db = get_db()
    # schema.sql should be in the root directory
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()
    print("Database initialized.")

@app.cli.command('init-db')
def init_db_command():
    """Flask CLI command to initialize the database."""
    init_db()
    print('Initialized the database.')

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
                    db = get_db()
                    db.execute(
                        "INSERT INTO users (name, email, image_data, detected_emotion) VALUES (?, ?, ?, ?)",
                        (name, email, image_data_for_db, emotion_label)
                    )
                    db.commit()
                except Exception as e:
                    print(f"Database save error: {e}")
                    error_text = "Prediction complete, but failed to save data to DB."

    return render_template('index.html', prediction_text=prediction_text, error_text=error_text)

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', error_text="File is too large. Max size is 5 MB."), 413

# --- Main Execution ---
if __name__ == '__main__':
    # We no longer init_db here. We use the 'flask init-db' command.
    app.run(debug=True)

