Drone Navigator’s Telemetry System

Real-Time Object Detection for Rescue Operations using YOLOv8 and OpenCV 

Description

This project implements a real-time drone-based object detection system designed for search-and-rescue operations. It integrates Deep Learning with Computer Vision to simulate how an autonomous drone detects, tracks, and locks onto targets such as people or vehicles from an aerial perspective.

Built as part of the Winter Training on Space Science and Technology conducted by India Space Academy, the system uses a pre-trained YOLOv8 model for detection and OpenCV for real-time video processing and visual overlays.

The project demonstrates how onboard vision systems enable drones to perceive their environment, filter detections intelligently, and provide actionable telemetry feedback through a dynamic Heads-Up Display (HUD).

Project Objective

The primary goal of this project is to integrate Deep Learning with Computer Vision to build a real-time drone telemetry interface capable of:

.Detecting objects using YOLOv8

.Filtering detections using confidence thresholds

.Calculating bounding box center coordinates using NumPy

.Applying intelligent logic for “Target Lock”

.Displaying real-time FPS and simulated distance metrics

Technologies Used

.Python

.OpenCV

.NumPy

.Ultralytics YOLOv8

.VS Code

Model Used

yolov8n.pt

.Video Input Handling

Captures video using cv2.VideoCapture()

Supports webcam or video file input

Processes frames continuously until user exits


. Object Detection (YOLOv8 Inference)

For each frame:

Runs YOLOv8 model inference

Extracts bounding box coordinates:
(x1, y1, x2, y2)

Retrieves confidence score

Filters detections using a confidence threshold (e.g., 0.6)


. Intelligence Filtering Logic

To simulate a smart targeting system:

Uses NumPy to calculate bounding box center:

Center_x = (x1 + x2) / 2

If center lies within the middle 20% of the screen:

Bounding box turns Red (Target Locked)

Otherwise:

Bounding box remains Green (Scanning Mode)

This mimics drone-based autonomous targeting logic.


. Visual Telemetry Overlay

The interface includes:

Bounding boxes using cv2.rectangle()

Object labels and confidence score

Simulated distance estimation using bounding box height

Live FPS counter

Drone-style HUD overlay


.Performance Optimization

Uses NumPy vectorized operations for bounding box calculations

Resizes frames to 640 × 640 for real-time inference

Maintains near real-time processing capability (target: 30 FPS)

Ensures clean shutdown using:

cap.release()

cv2.destroyAllWindows()

