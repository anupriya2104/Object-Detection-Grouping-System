# Retail Product Detection and Grouping System

## Overview

This project is a microservice-based computer vision pipeline designed to detect retail products from shelf images, generate feature embeddings, group visually similar products, and visualize the final results.

The system uses object detection and image similarity techniques to automatically identify and cluster products present in retail shelf images.

---

## Architecture

The application follows a microservice architecture consisting of four independent services:

### 1. Detector Service

* Receives shelf images.
* Detects products using YOLOv8.
* Crops individual product images.
* Stores cropped products for further processing.

### 2. Grouping Service

* Generates embeddings for cropped products using CLIP.
* Performs similarity analysis.
* Groups similar products using K-Means clustering.
* Produces structured grouping results.

### 3. Visualization Service

* Creates visual outputs showing grouped products.
* Generates cluster-wise product displays.
* Provides an easy-to-understand representation of results.

### 4. Gateway Service

* Acts as the central entry point.
* Coordinates communication between services.
* Exposes APIs for processing requests.

---

## Technology Stack

### Programming Language

* Python

### Frameworks

* Flask
* FastAPI

### Machine Learning & Computer Vision

* YOLOv8
* CLIP
* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* OpenCV

---

## Workflow

1. User uploads a retail shelf image.
2. Gateway forwards the image to the Detector Service.
3. Detector Service identifies products and creates cropped images.
4. Cropped images are sent to the Grouping Service.
5. CLIP generates feature embeddings for each product.
6. K-Means groups visually similar products.
7. Visualization Service creates grouped output images.
8. Results are returned to the user.

---

## Folder Structure

```text
project-root/
│
├── detector_service/
│   ├── app.py
│   ├── model/
│   └── outputs/
│
├── grouping_service/
│   ├── app.py
│   ├── clustering.py
│   └── embeddings/
│
├── visualization_service/
│   ├── app.py
│   └── visualization.py
│
├── gateway/
│   └── app.py
│
├── uploads/
├── outputs/
├── requirements.txt
├── test_api.py
├── test_visualization.py
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd retail-product-grouping
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Services

### Start Detector Service

```bash
cd detector_service
python app.py
```

### Start Grouping Service

```bash
cd grouping_service
python app.py
```

### Start Visualization Service

```bash
cd visualization_service
python app.py
```

### Start Gateway Service

```bash
cd gateway
python app.py
```

---

## API Usage

### Upload Shelf Image

```http
POST /process
```

#### Request

```bash
curl -X POST \
-F "image=@shelf.jpg" \
http://localhost:5000/process
```

#### Response

```json
{
  "status": "success",
  "clusters": 5,
  "visualization": "output.png"
}
```

---

## Features

* Product detection using YOLOv8
* Product similarity analysis using CLIP embeddings
* Automatic grouping using K-Means clustering
* Microservice-based architecture
* REST API support
* Scalable and modular design
* Visualization of grouped products

---

## Future Improvements

* Support for real-time video streams
* Dynamic cluster selection
* Distributed deployment using Docker and Kubernetes
* Product inventory analytics
* Retail shelf compliance monitoring
