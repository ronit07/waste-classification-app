"""
Standalone inference helper.
Usage: python model/predict.py path/to/image.jpg
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import sys

CLASSES  = ['recyclable', 'organic', 'landfill']
IMG_SIZE = (224, 224)

DESCRIPTIONS = {
    'recyclable': 'Place in the blue recycling bin. Clean and dry before recycling.',
    'organic':    'Place in the green compost bin. Great for food scraps and garden waste.',
    'landfill':   'Place in the general waste bin. This item cannot be recycled or composted.',
}

def load_model(path='model/waste_model.h5'):
    return tf.keras.models.load_model(path)

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image_path, model=None):
    if model is None:
        model = load_model()
    x     = preprocess(image_path)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        'category':    CLASSES[idx],
        'confidence':  round(float(probs[idx]), 4),
        'description': DESCRIPTIONS[CLASSES[idx]],
        'all_probs': {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python model/predict.py <image_path>")
        sys.exit(1)
    result = predict(path)
    print(f"Category:    {result['category'].upper()}")
    print(f"Confidence:  {result['confidence']*100:.1f}%")
    print(f"Description: {result['description']}")
