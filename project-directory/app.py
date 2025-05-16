from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import re

app = Flask(__name__)
model = load_model('digit_model.h5')

# Обработка изображения из base64 (канвас)
def preprocess_base64(img_data):
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    decoded = base64.b64decode(img_str)
    return preprocess_image(decoded)

# Обработка изображения из файла
def preprocess_image(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert('L')
    img = img.resize((28, 28))
    img_arr = np.array(img) / 255.0
    img_arr = 1 - img_arr  # инверсия
    return img_arr.reshape(1, 28, 28)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_from_canvas():
    data = request.get_json()
    img = preprocess_base64(data['image'])
    pred = model.predict(img)
    return jsonify({'prediction': int(np.argmax(pred))})

@app.route('/upload', methods=['POST'])
def predict_from_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    file = request.files['file']
    img = preprocess_image(file.read())
    pred = model.predict(img)
    return jsonify({'prediction': int(np.argmax(pred))})

if __name__ == '__main__':
    app.run(debug=True)
