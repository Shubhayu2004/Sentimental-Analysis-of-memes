from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import TextVectorization
import re
import string
import pickle
from PIL import Image
import pytesseract

app = Flask(__name__)

# --- Load Model and Vectorizer ---
MODEL_PATH = "best_model.keras"
combined_model = load_model(MODEL_PATH)

vocab_size = 100000
sequence_length = 50
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation"
)
with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vectorize_layer.set_vocabulary(vocab)

def standardize_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'.com', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_image(img_file):
    img = image.load_img(img_file, target_size=(100, 100, 3))
    img = image.img_to_array(img) / 255.0
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image.'}), 400

    img_file = request.files['image']

    # OCR: Extract text from image
    pil_img = Image.open(img_file)
    extracted_text = pytesseract.image_to_string(pil_img)
    text = standardize_text(extracted_text)
    text_vec = vectorize_layer([text])
    text_vec = np.array(text_vec).astype('int32')

    # Preprocess image for model
    img_file.seek(0)  # Reset file pointer after PIL read
    img = preprocess_image(img_file)
    img = np.expand_dims(img, axis=0).astype('float32')

    preds = combined_model.predict({'image_input': img, 'text': text_vec})
    preds = preds[0]

    response = {
        'humour': float(preds[0]),
        'sarcasm': float(preds[1]),
        'offensive': float(preds[2]),
        'motivational': float(preds[3]),
        'extracted_text': extracted_text.strip()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)