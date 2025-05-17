# Cat and Dog Breed Predictor

A web application that predicts cat and dog breeds from uploaded images using machine learning. Built with Flask and powered by MobileNetV2 architecture for efficient and accurate breed classification. The application combines modern web technologies with deep learning to provide real-time predictions.

## Features

- 🐱 Cat and dog breed prediction with confidence scores
- 🖼️ Real-time image preview
- 🎨 Modern, responsive UI with cute animations
- ✨ Confidence threshold filtering (85%)
- 🔄 Loading indicators and error handling

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <project-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure the model file `my_model_55epochs.keras` is in the correct location.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── model/                 # ML model directory
├── static/               # Static assets
│   ├── style.css         # CSS styles
│   ├── cat-doodle.svg    # SVG illustrations
│   ├── dog-doodle.svg
│   └── uploads/          # Upload directory
├── templates/            # HTML templates
│   └── index.html
└── class_indices.xlsx    # Breed mapping file
```

## Technologies Used

- Backend: Flask (Python)
- ML Framework: TensorFlow/Keras
- Model Architecture: MobileNetV2 (pre-trained and fine-tuned for breed classification)
- Frontend: HTML5, CSS3
- Font: Fredoka (Google Fonts)
- Styling: Custom CSS with animations

## Notes

- The model requires uploaded images to be in a supported format (jpg, jpeg, png)
- Predictions with confidence below 85% will not be shown
- The application uses a pre-trained model optimized for cat and dog breed recognition 