from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import torch

app = Flask(__name__)

# Load CLIP model once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@app.route("/group", methods=["POST"])
def group_products():

    try:
        image_path = request.json["image_path"]
        detections = request.json["detections"]

        image = Image.open(image_path).convert("RGB")

        crops = []

        # Crop detected products
        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            crop = image.crop((x1, y1, x2, y2))

            crops.append(crop)

        # Edge cases
        if len(crops) == 0:
            return jsonify({
                "groups": []
            })

        if len(crops) == 1:
            return jsonify({
                "groups": [0]
            })

        # Create CLIP inputs
        inputs = processor(
            images=crops,
            return_tensors="pt",
            padding=True
        )

        # Use only image tensor
        vision_inputs = {
            "pixel_values": inputs["pixel_values"]
        }

        # Extract image features
        with torch.no_grad():
            outputs = model.vision_model(**vision_inputs)

        features = outputs.pooler_output.cpu().numpy()

        # Cluster products
        n_clusters = min(6, len(crops))

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(features)

        return jsonify({
            "groups": labels.tolist()
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(port=5002, debug=True)