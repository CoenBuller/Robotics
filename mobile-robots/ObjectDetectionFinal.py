# background_detection.py
import cv2
import numpy as np


def learn_background(cap, duration_seconds=10):
    """
    Feed frames to MOG2 for a fixed duration to build a stable background model.
    Returns the trained background subtractor.
    """
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=False
    )

    start = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bg.apply(frame)

        elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        remaining = int(duration_seconds - elapsed)

        display = frame.copy()
        cv2.putText(display, f"Learning background: {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Setup", display)
        cv2.waitKey(1)

        if elapsed >= duration_seconds:
            break

    print("Background learned. Place your object in the scene.")
    return bg


def detect_object(cap, bg, min_area=500):
    """
    Apply background subtraction to find the largest new foreground blob.
    Returns (frame, bbox) where bbox is the bounding rect of the object
    """

    ret, frame = cap.read()
    if not ret:
        return None, None

    # Detect new object in the learned background frame
    mask = bg.apply(frame, learningRate=0)

    # Make mask connecting with smoother edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours around the objects mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not contours:
        return frame, None # No object dected -> Fail

    H, W = frame.shape[:2]
    def score(c):  # Candidate contour score is based on size of contour and distance to center
        area=cv2.contourArea(c)
        x,y,w,h=cv2.boundingRect(c)

        cx,cy=x+w/2, y+h/2
        dist_center=((cx-W/2)**2 +(cy-H/2)**2)**0.5

        return area/(1+dist_center)
    
    largest = max(contours, key=score)
    x, y, w, h = cv2.boundingRect(largest)

    bbox = (int(x), int(y), int(w), int(h))
    return frame, bbox


def ObjectDetection(cam, observation_duration=10, min_area=500, scale=0.9):
    """Returns (bbox, scale) once a stable detection is found."""
    bg = learn_background(cam, duration_seconds=observation_duration)

    print("Detecting object — press Q to lock in.")
    bbox = None
    while True:
        frame, bbox = detect_object(cam, bg, min_area=min_area)
        if frame is None:
            break

        if bbox is not None: # Draw bounding box around detected object
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Object detected", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return bbox


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    bbox = ObjectDetection(cap, observation_duration=5, min_area=500)
    cap.release()
    cv2.destroyAllWindows()
    print("Final bbox:", bbox)
