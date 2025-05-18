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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

def preprocess_image(img):
    """Preprocess the image for model prediction."""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        logger.debug(f"Original image size: {img.size}")
        
        # Resize image
        img = img.resize((224, 224))
        logger.debug("Image resized to 224x224")
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32')
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Final input shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def get_prediction(img_array):
    """Get prediction from preprocessed image array."""
    try:
        # Make prediction
        logger.debug("Making prediction...")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'})
            
            # Read the image file
            img = Image.open(file.stream)
            
        elif 'url' in request.form:
            # Handle URL upload
            url = request.form['url']
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
            except Exception as e:
                logger.error(f"Error fetching URL: {str(e)}")
                return jsonify({'error': 'Unable to fetch image from URL. Please check the URL or try uploading the file directly.'})
        else:
            return jsonify({'error': 'No file or URL provided'})

        # Process the image and get prediction
        try:
            img_array = preprocess_image(img)
            prediction = get_prediction(img_array)
            
            return jsonify({
                'prediction': prediction
            })
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Error processing image. Please try another image.'})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
