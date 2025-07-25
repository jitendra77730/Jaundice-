from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("jaundice_model_mobilenetv2.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Jaundice" if prediction > 0.5 else "Healthy"
    probability = round(float(prediction), 3)
    return label, probability

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        f = request.files["file"]
        file_path = os.path.join("static", f.filename)
        os.makedirs("static", exist_ok=True)
        f.save(file_path)
        label, prob = predict_image(file_path)
        result = f"Result: {label} (Probability: {prob})"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
