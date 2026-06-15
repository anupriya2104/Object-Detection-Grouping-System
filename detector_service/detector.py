from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests

app = Flask(__name__)

# Better detector than yolov8n
model = YOLO("yolov8s.pt")


@app.route("/detect", methods=["POST"])
def detect():

    image_path = request.json["image_path"]

    results = model(
        image_path,
        conf=0.10,
        iou=0.30
    )

    detections = []

    image_height = results[0].orig_shape[0]
    image_width = results[0].orig_shape[1]

    image_area = image_width * image_height

    for box in results[0].boxes:

        conf = float(box.conf[0])

        if conf < 0.20:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1

        area = width * height

        # Reject huge detections
        if area > 0.15 * image_area:
            continue

        detections.append({
            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "confidence": round(conf, 3)
        })

    # Call Grouping Service
    group_response = requests.post(
        "http://127.0.0.1:5002/group",
        json={
            "image_path": image_path,
            "detections": detections
        }
    )

    group_data = group_response.json()

    if "groups" not in group_data:
        return jsonify(group_data)

    groups = group_data["groups"]

    # Call Visualization Service
    visual_response = requests.post(
        "http://127.0.0.1:5003/visualize",
        json={
            "image_path": image_path,
            "detections": detections,
            "groups": groups
        }
    )

    visual_data = visual_response.json()

    return jsonify({
        "detections": detections,
        "groups": groups,
        "output_image": visual_data["output_image"]
    })


if __name__ == "__main__":
    app.run(
        port=5001,
        debug=True
    )