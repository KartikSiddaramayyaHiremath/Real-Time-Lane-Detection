# Real-Time-Lane-Detection
“Real-time lane marking detection for road images using color filtering, morphological operations, and polynomial fitting to highlight white and yellow lanes.”
---

## Project Description

**Lane Marking Detection (Image-Based)** is a real-time computer vision project that identifies and highlights road lane markings in images. Using classical techniques like **color filtering, morphological operations, and Hough Transform**, the system can detect **white and yellow lane lines**, handle **curved lanes**, and overlay **dark green lane lines** on a semi-transparent **region of interest (ROI)**.

This project is ideal for learning **image processing for autonomous driving applications** without requiring deep learning models. It works efficiently on standard images and saves processed outputs automatically.

**Key Features:**

- Detects straight and curved lane markings in road images
- Uses HLS and HSV color filtering for robustness to lighting
- Noise reduction using morphological operations
- Configurable ROI for focused processing
- Outputs are stored automatically for easy visualization

---

## Folder Structure

Lane-Marking-Detection/
│
├── src/
│ ├── lane_detection.py
│
├── samples/
│ └── images/ 
│
├── outputs/
│ └── processed_images/ 
│
├── README.md
└── requirements.txt



