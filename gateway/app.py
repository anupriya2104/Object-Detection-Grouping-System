from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(image_path)

    response = requests.post(
        "http://127.0.0.1:5001/detect",
        json={
            "image_path": image_path
        }
    )

    print(response.text)

    return response.text

if __name__ == "__main__":
    app.run(port=5000, debug=True)