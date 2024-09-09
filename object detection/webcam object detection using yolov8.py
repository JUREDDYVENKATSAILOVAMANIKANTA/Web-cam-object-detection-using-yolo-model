#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install ultralytics


# In[ ]:


import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8n.pt (nano), or switch to yolov8s.pt, yolov8m.pt for larger models

# Open video stream (replace '0' with a video file path if you are using a pre-recorded video)
video_path = 0  # Use 0 for webcam or provide a file path for a video
cap = cv2.VideoCapture(video_path)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if the video ends or there's an error

    # Perform object detection using YOLOv8
    results = model(frame)

    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()  # This will draw bounding boxes and labels on the frame

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




