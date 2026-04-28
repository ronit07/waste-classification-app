import os, io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

CLASSES  = ['recyclable', 'organic', 'landfill']
IMG_SIZE = (224, 224)

DESCRIPTIONS = {
    'recyclable': 'Place in the blue recycling bin. Clean and dry before recycling.',
    'organic':    'Place in the green compost bin. Great for food scraps and garden waste.',
    'landfill':   'Place in the general waste bin. This item cannot be recycled or composted.',
}

COLORS = {'recyclable': '#2196F3', 'organic': '#4CAF50', 'landfill': '#9E9E9E'}

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'waste_model.h5')
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize(IMG_SIZE)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file_bytes = request.files['image'].read()
    try:
        x     = preprocess(file_bytes)
        probs = get_model().predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        cat   = CLASSES[idx]

        return jsonify({
            'category':    cat,
            'confidence':  round(float(probs[idx]), 4),
            'description': DESCRIPTIONS[cat],
            'color':       COLORS[cat],
            'all_probs':   {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
