import requests

response = requests.post(
    "http://127.0.0.1:5003/visualize",
    json={
        "image_path": "test.jpg",
        "detections": [
            {
                "bbox": [124,103,149,188]
            },
            {
                "bbox": [182,103,215,190]
            }
        ],
        "groups": [0,1]
    }
)

print(response.json())