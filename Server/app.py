from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# === CONFIGURATION ===
MODEL_PATH = r"C:\Users\tluke\Desktop\CoffeGuard\Models\best_model.h5"  # or .tflite
LABELS = ['healthy', 'miner', 'phoma', 'rust']
IMAGE_SIZE = (224, 224)

# === MODEL LOADING ===
model = None
is_tflite = MODEL_PATH.lower().endswith(".tflite")

try:
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Loaded TensorFlow Lite model successfully.")
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Loaded Keras model successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise SystemExit(e)


# === IMAGE PREPROCESSING ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# === PREDICTION ROUTE ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(file)
        img_array = preprocess_image(image)

        if is_tflite:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            preds = model.predict(img_array)[0]

        label_idx = int(np.argmax(preds))
        result = {
            "label": LABELS[label_idx],
            "confidence": round(float(np.max(preds)) * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# === RUN SERVER ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
