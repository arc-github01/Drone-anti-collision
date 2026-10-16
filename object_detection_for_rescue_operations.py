"""
------------------------------------------------------------
IndiaSpaceAcademy  DRONE NAVIGATOR TELEMETRYSYSTEM
------------------------------------------------------------

Project Name : Object Detection for Rescue Operations.
Author       : Harsha Jaiswal
Chosen Model : YOLOv8 Small (yolov8s.pt)

Description:
This project implements real-time object detection using 
YOLOv8 and simulates a drone rescue target locking system 
with distance estimation and HUD overlay.

------------------------------------------------------------
"""

import cv2
import numpy as np
import math
import time
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
# "Object Detection for Rescue Operations" 
MODEL_NAME = 'yolov8s.pt'  # Small model — better accuracy, still fast
CONF_THRESHOLD = 0.6       # Filter out "ghost" detections 
TARGET_LOCK_ZONE = 0.20    # Middle 20% of screen [cite: 43]

def process_drone_feed(source=0):
    """
    Main loop for the Drone Navigator's Telemetry System.
    source: 0 for webcam, or path to video file string.
    """
    #------------------------------
    # 1. Initialization [cite: 70]
    #------------------------------
    # Resolve file path BEFORE loading YOLO (YOLO can change working directory)
    if isinstance(source, str):
        source = os.path.abspath(source)
        print(f"Video source: {source}")
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the "Lock Zone" boundaries (Middle 20%)
    # Logic: 40% left padding, 20% middle, 40% right padding.
    lock_zone_start = frame_width * 0.4
    lock_zone_end = frame_width * 0.6

    print("Drone System Online. Press 'q' to exit.")
    #----------------------------
    # 2. Main Loop [cite: 71]
    #-------------------------
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Optimization: Resize to 640x640 if lagging [cite: 56]
        # frame = cv2.resize(frame, (640, 640)) 

        # 3. Inference: Run YOLOv8 on the frame [cite: 37]
        results = model(frame, verbose=False)
        result = results[0] # Get first result

        # 4. Vectorization & Filtering [cite: 54]
        # Extract boxes (xyxy), confidence (conf), and class (cls) as NumPy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordinates
        confs = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy() # Class IDs

        # Filter by Confidence Threshold > 0.6 
        valid_indices = np.where(confs > CONF_THRESHOLD)[0]

        # Iterate only through valid detections
        for i in valid_indices:
            # Extract coordinates
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate Center Point [cite: 42]
            center_x = (x1 + x2) / 2
            
            # Calculate Box Height for Distance Simulation 
            box_height = y2 - y1
            
            # --- INTELLIGENCE LOGIC ---
            # Check if center is in the middle 20% [cite: 43]
            if lock_zone_start < center_x < lock_zone_end:
                color = (0, 0, 255) # Red for Target Lock
                status = "LOCKED"
            else:
                color = (0, 255, 0) # Green for Scanning [cite: 44]
                status = "SCANNING"

            # Distance Simulation Logic (Simple Inverse Mapping)
            # Smaller box = Larger distance. 
            # This is a heuristic approximation for the HUD.
            estimated_dist_meters = round(3000 / (box_height + 1), 1) 

            # --- VISUAL OVERLAY ---
            # Draw Bounding Box [cite: 46]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with Name and Distance 
            class_name = model.names[int(classes[i])]
            label = f"{class_name} | {estimated_dist_meters}m"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw Center Point for visual reference
            cv2.circle(frame, (int(center_x), int((y1+y2)/2)), 4, color, -1)

        # Draw the Lock Zone Lines (HUD)
        cv2.line(frame, (int(lock_zone_start), 0), (int(lock_zone_start), frame_height), (255, 255, 0), 1)
        cv2.line(frame, (int(lock_zone_end), 0), (int(lock_zone_end), frame_height), (255, 255, 0), 1)
        cv2.putText(frame, "TARGET ZONE", (int(lock_zone_start) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Calculate and Display FPS 
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the Drone HUD
        cv2.imshow("Drone Rescue HUD", frame)

        # Loop Control: Break on 'q' [cite: 51]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #------------------------------------------
    # 5. Shutdown: Release resources [cite: 52]
    #------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can replace 0 with a video file path like "forest_flight.mp4"
    process_drone_feed("testv2.mp4")
