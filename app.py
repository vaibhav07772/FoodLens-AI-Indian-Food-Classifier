"""
app.py
------
Flask web application for the Indian Food classifier. Users upload a
food photo; the image is pre-processed and passed through the trained
CNN, and the response includes:

  - predicted dish name (English + Hindi) and confidence score
  - top-3 alternatives
  - nutrition facts (calories, protein, carbs, fat, calcium per 100g)
  - a Vegetarian / Non-Vegetarian badge
  - an approximate "walk it off" calorie-burn estimate

Run:
    python app.py

Then open http://127.0.0.1:5000 in a browser.

If no trained model is found at model/food_classifier.h5, the app still
runs in DEMO MODE so the interface can be explored end-to-end: it
returns a clearly-labelled simulated prediction instead of a real one.
Train a real model with train.py (or train_on_colab.ipynb) first.
"""

import io
import os
import random

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

from nutrition_data import get_nutrition, FOOD_NUTRITION, calories_to_walk_minutes

MODEL_PATH = "model/food_classifier.h5"
LABELS_PATH = "model/labels.txt"
IMAGE_SIZE = (224, 224)

MODEL_BACKBONE = "efficientnetb0"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

DEMO_LABELS = list(FOOD_NUTRITION.keys())

_model = None
_labels = None


def load_model_if_available():
    """Lazily loads the trained Keras model and label map, if present."""
    global _model, _labels
    if _model is not None:
        return

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            from tensorflow.keras.models import load_model
            _model = load_model(MODEL_PATH, compile=False)
            with open(LABELS_PATH) as f:
                _labels = {str(i): name.strip() for i, name in enumerate(f) if name.strip()}
            print(f"✅ Loaded trained model from {MODEL_PATH} ({len(_labels)} classes)")
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            print("→ Falling back to DEMO MODE")
            _model = None
            _labels = None
    else:
        print("No trained model found — running in DEMO MODE.")
        _model = None
        _labels = None


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    if MODEL_BACKBONE == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    array = np.asarray(img, dtype="float32")
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict(file_bytes: bytes):
    """Returns (predicted_label, confidence_percent, top3_list, demo_mode)."""
    load_model_if_available()

    if _model is not None:
        batch = preprocess_image(file_bytes)
        probs = _model.predict(batch, verbose=0)[0]
        ranked = sorted(
            ((_labels[str(i)], float(p)) for i, p in enumerate(probs)),
            key=lambda x: x[1],
            reverse=True,
        )
        top_label, top_prob = ranked[0]
        top3 = [{"label": lbl, "confidence": round(p * 100, 2)} for lbl, p in ranked[:3]]
        return top_label, round(top_prob * 100, 2), top3, False

    # ---- Demo mode ----
    random.shuffle(DEMO_LABELS)
    top_label = DEMO_LABELS[0]
    top_prob = round(random.uniform(85, 99), 2)
    remaining = DEMO_LABELS[1:3]
    top3 = [{"label": top_label, "confidence": top_prob}]
    leftover = 100 - top_prob
    for lbl in remaining:
        share = round(leftover / (len(remaining) + 1), 2)
        top3.append({"label": lbl, "confidence": share})
    return top_label, top_prob, top3, True


def format_label(label: str) -> str:
    return label.replace("_", " ").title()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    allowed_ext = (".jpg", ".jpeg", ".png")
    if not file.filename.lower().endswith(allowed_ext):
        return jsonify({"error": "Please upload a JPG or PNG image."}), 400

    try:
        file_bytes = file.read()
        label, confidence, top3, demo_mode = predict(file_bytes)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 500

    nutrition = get_nutrition(label)
    walk_minutes = calories_to_walk_minutes(nutrition.get("calories"), portion_g=150)

    return jsonify({
        "label": format_label(label),
        "label_hindi": nutrition.get("name_hindi"),
        "confidence": confidence,
        "top3": [{"label": format_label(t["label"]), "confidence": t["confidence"]} for t in top3],
        "nutrition": nutrition,
        "diet": nutrition.get("diet"),
        "walk_minutes": walk_minutes,
        "demo_mode": demo_mode,
    })


if __name__ == "__main__":
    load_model_if_available()
    app.run(debug=True)