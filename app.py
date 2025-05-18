from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import logging
import requests
from io import BytesIO
from functools import partial
import concurrent.futures
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_IMAGE_SIZE = (1920, 1080)  # Maximum allowed image dimensions
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size
PREDICTION_TIMEOUT = 10  # Timeout for prediction in seconds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the pre-trained model and class indices
try:
    logger.info("Loading model from: model/my_model_55epochs.keras")
    model = tf.keras.models.load_model('model/my_model_55epochs.keras')
    logger.info("Model loaded successfully")
    
    logger.info("Loading class indices from: class_indices.xlsx")
    df = pd.read_excel("class_indices.xlsx")
    # Create mapping from index to breed name using correct column names
    breed_names = dict(zip(df['Class Index'], df['Class Name']))
    logger.info(f"Loaded {len(breed_names)} breed names")
    logger.debug(f"Available breeds: {breed_names}")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

def check_image_size(img):
    """Check if image dimensions are within acceptable limits."""
    if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
        aspect_ratio = img.size[0] / img.size[1]
        if aspect_ratio > MAX_IMAGE_SIZE[0] / MAX_IMAGE_SIZE[1]:
            new_size = (MAX_IMAGE_SIZE[0], int(MAX_IMAGE_SIZE[0] / aspect_ratio))
        else:
            new_size = (int(MAX_IMAGE_SIZE[1] * aspect_ratio), MAX_IMAGE_SIZE[1])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def preprocess_image(img):
    """Preprocess the image for model prediction."""
    try:
        start_time = time.time()
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Check and resize large images
        img = check_image_size(img)
        logger.debug(f"Image size after check: {img.size}")
        
        # Resize image
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.debug(f"Preprocessing took {time.time() - start_time:.2f} seconds")
        return img_array
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def get_prediction(img_array):
    """Get prediction from preprocessed image array."""
    try:
        start_time = time.time()
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        logger.info(f"Prediction took {time.time() - start_time:.2f} seconds")
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
        
        # Check confidence threshold
        if confidence < 0.85:  # 85% threshold
            logger.info(f"Low confidence prediction: {confidence:.2%}")
            return "Hm it's hard to guess. Please try to upload a better picture"
        
        # Get breed name for high confidence predictions
        breed = breed_names.get(predicted_class, "Unknown")
        logger.info(f"Predicted breed: {breed}")
        
        return f"{breed} (Confidence: {confidence:.2%})"
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def process_image_with_timeout(img):
    """Process image with timeout."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            future = executor.submit(preprocess_image, img)
            img_array = future.result(timeout=PREDICTION_TIMEOUT/2)
            
            future = executor.submit(get_prediction, img_array)
            return future.result(timeout=PREDICTION_TIMEOUT/2)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Image processing took too long. Please try a different image.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'})
            
            try:
                img = Image.open(file.stream)
            except Exception as e:
                logger.error(f"Error opening image file: {str(e)}")
                return jsonify({'error': 'Unable to open image file. The file might be corrupted.'})
            
        elif 'url' in request.form:
            url = request.form['url']
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content))
            except requests.Timeout:
                return jsonify({'error': 'URL request timed out. Please try a different URL or upload the file directly.'})
            except Exception as e:
                logger.error(f"Error fetching URL: {str(e)}")
                return jsonify({'error': 'Unable to fetch image from URL. Please check the URL or try uploading the file directly.'})
        else:
            return jsonify({'error': 'No file or URL provided'})

        try:
            prediction = process_image_with_timeout(img)
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return jsonify({'prediction': prediction})
        except TimeoutError as e:
            return jsonify({'error': str(e)})
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Error processing image. Please try another image.'})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
