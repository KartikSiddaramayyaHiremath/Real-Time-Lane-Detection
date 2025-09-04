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

## Requirements

- **Python 3.8 or higher**
- **Libraries**:
  - `opencv-python` (for image processing and lane detection)
  - `numpy` (for numerical operations and polynomial fitting)
 
## Workflow

-Load road image
-Apply color filtering (HLS / HSV)
-Apply morphological operations
-Detect edges using Canny
-Extract ROI (Region of Interest)
-Detect lines using Hough Transform
-Draw polynomial-fitted lane lines & highlight ROI
-Save the final output

## Color Filtering

Color filtering is used to isolate white and yellow lane markings from road images. By converting the image into the HLS color space, the system applies thresholding to highlight lanes while ignoring irrelevant background details like road texture or vehicles.

## Morphology 

Morphological operations (like closing) are applied to the filtered image to reduce noise and fill small gaps in the detected lane regions. This ensures the lane lines appear cleaner, more continuous, and easier to detect in the next processing steps.

## **Applications**

- Driver Assistance Systems (ADAS)
- Fundamentals of Autonomous Driving Vision
- Educational project for Computer Vision & OpenCV learners

## **Future Scope**

- Extend to video-based real-time detection
- Improve robustness under rain, night, and shadow conditions
- Integrate with deep learning for advanced lane classification




