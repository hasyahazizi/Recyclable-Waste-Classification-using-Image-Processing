from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow import keras 
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing import image as k_image
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input

print("Tensorflow version: ", tf.__version__)

app = Flask(__name__)

# Load model
model = keras.models.load_model('densenet_model_v3.h5', compile=False)

# Set model to inference mode
model.trainable = False

# Class names mapping
CLASS_NAMES = [
    'cardboard', 'glass', 'metal', 
    'paper', 'plastic', 'trash'
]

# Specify upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/classify', methods=['GET'])
def predict():
    return render_template("classifier.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess the image
        img = k_image.load_img(file_path, target_size=(224, 224))
        img_array = k_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        # Render the results in a new template
        return render_template("result.html", predicted_class=CLASS_NAMES[predicted_index],
                               confidence=confidence, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)