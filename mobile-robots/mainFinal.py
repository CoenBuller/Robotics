import numpy as np
import cv2
from ObjectDetectionFinal import ObjectDetection
from TrackerFinal import Tracker
# from pi_camera import PiCam

# Setup camera
cam = cv2.VideoCapture(0) # Use the first camera
scale = 1 
bbox = ObjectDetection(cam, observation_duration=3, min_area=500, scale=scale)
if bbox is not None:
	print("Lockbbox:" ,{bbox})
	Tracker(cam, bbox,drive_motors=False, scale=0.5)

cam.release()
cv2.destroyAllWindows()
