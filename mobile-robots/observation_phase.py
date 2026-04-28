import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Background subtractor (for motion detection)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# Storage for averaging
bboxes = []
histograms = []

print("Observation phase started...")
frame_count = 0
MAX_FRAMES = 60  # ~2-3 seconds depending on FPS

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()  # Fixed: was [cap.read](http://cap.read)()
    if not ret:
        break

    # Apply background subtraction
    fgmask = bg_subtractor.apply(frame)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take largest contour (likely the object)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:  # filter small noise
            x, y, w, h = cv2.boundingRect(largest)

            # Save bbox
            bboxes.append((x, y, w, h))

            # Extract ROI
            roi = frame[y:y+h, x:x+w]

            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            histograms.append(hist)

            # Draw for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Observation", frame)
    cv2.imshow("FG Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# --- Post-processing (averaging) ---
if not bboxes:
    raise Exception("No object detected during observation phase.")

# Average bounding box
avg_bbox = np.mean(bboxes, axis=0).astype(int)
x, y, w, h = avg_bbox

# Average histogram
avg_hist = np.mean(histograms, axis=0)

print("Final bbox:", avg_bbox)


# # --- Initialize tracker ---
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()  # Fixed: was [cap.read](http://cap.read)()
# tracker = cv2.legacy.TrackerCSRT_create()
# tracker.init(frame, (x, y, w, h))
# target_size = w * h

# print("Tracker initialized. Starting tracking...")

# # --- Tracking loop ---
# while True:
#     ret, frame = cap.read()  # Fixed: was [cap.read](http://cap.read)()
#     if not ret:
#         break

#     success, bbox = tracker.update(frame)

#     if success:
#         x, y, w, h = map(int, bbox)

#         # Draw tracked object
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # --- Control signals ---
#         frame_center = frame.shape[1] // 2
#         object_center = x + w // 2
#         error_x = object_center - frame_center
#         size = w * h
#         error_size = target_size - size

#         print(f"Steering error: {error_x}, Distance error: {error_size}")
#     else:
#         cv2.putText(frame, "Tracking lost!", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

cap.release()
cv2.destroyAllWindows()