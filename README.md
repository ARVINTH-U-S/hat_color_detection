# Hat Color Detection System

## Overview
This project detects **hat colors worn by people in a video stream** using **YOLOv8** for object detection and **OpenCV** for image processing.

It performs:
- Person detection and tracking
- Hat detection
- Hat color classification using HSV thresholds
- Stable color assignment using frame confirmation logic

---

## How It Works

### 1. Object Detection
- Uses **YOLOv8 (`yolov8m.pt`)** model
- Detects:
  - `person` (class 0)
  - `hat` (class 1)

### 2. Tracking
- Uses YOLO tracking (`model.track`)
- Assigns unique `track_id` for each person

### 3. Hat-Color Mapping
- Checks if hat bounding box lies inside a person bounding box
- Assigns detected hat color to that person

### 4. Color Detection
- Extracts a small region from the center of the hat
- Converts to HSV color space
- Applies predefined color thresholds

### 5. Stability Logic
- Color must appear for **5 consecutive frames**
- Prevents flickering and false detection

---

## Project Structure

    hatcolor_lock_pcount 1.py
    color_detector.py
    README.md

---

## Requirements

```bash
pip install ultralytics opencv-python numpy pillow

---
