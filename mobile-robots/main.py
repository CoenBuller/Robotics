import numpy as np
import cv2
from ObjectDetection import ObjectDetection
from ColorShapeTracker import TrackerDetector


# Setup camera
cam = cv2.VideoCapture(0)

# Detect object using background substraction method and create a bounding box
bbox, template, hsv_ranges = ObjectDetection(cam, observation_duration=10, min_area=500, scale=1)

if bbox is not None:
    TrackerDetector(cam, template, bbox)
cam.release()
cv2.destroyAllWindows()
