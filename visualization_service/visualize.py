from flask import Flask, request, jsonify
import cv2
import os

app = Flask(__name__)

os.makedirs("outputs", exist_ok=True)

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255)
]


@app.route("/visualize", methods=["POST"])
def visualize():

    image_path = request.json["image_path"]
    detections = request.json["detections"]
    groups = request.json["groups"]

    image = cv2.imread(image_path)

    for det, grp in zip(detections, groups):

        x1, y1, x2, y2 = det["bbox"]

        color = COLORS[grp % len(COLORS)]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            f"G{grp}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    output_path = "outputs/result.jpg"

    cv2.imwrite(output_path, image)

    return jsonify({
        "output_image": output_path
    })


if __name__ == "__main__":
    app.run(port=5003, debug=True)