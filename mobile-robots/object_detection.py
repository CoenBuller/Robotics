import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
TARGET_CLASS = "bottle", "cup", "ball"

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

MIN_BBOX_HEIGHT = 80
MAX_BBOX_HEIGHT = 200
TURN_THRESHOLD  = 60


def detect_target(frame, model, target_class):
    results = model(frame, verbose=False)[0]

    best_box = None
    best_conf = 0.0

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf     = float(box.conf[0])

        if cls_name == target_class and conf > best_conf:
            best_conf = conf
            best_box  = box

    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = x2 - x1
    bh = y2 - y1

    return cx, cy, bw, bh


def compute_robot_command(cx, cy, bw, bh, frame_width):
    frame_cx = frame_width // 2
    error_x  = cx - frame_cx

    if error_x > TURN_THRESHOLD:
        return "TURN_RIGHT"
    elif error_x < -TURN_THRESHOLD:
        return "TURN_LEFT"

    if bh < MIN_BBOX_HEIGHT:
        return "MOVE_FORWARD"
    elif bh > MAX_BBOX_HEIGHT:
        return "STOP"
    else:
        return "MOVE_FORWARD"


def draw_overlay(frame, cx, cy, bw, bh, command):
    x1 = cx - bw // 2
    y1 = cy - bh // 2
    x2 = cx + bw // 2
    y2 = cy + bh // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.line(frame, (frame.shape[1] // 2, 0),
             (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
    cv2.putText(frame, f"CMD: {command}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"cx={cx} bh={bh}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_target(frame, model, TARGET_CLASS)

        if result:
            cx, cy, bw, bh = result
            command = compute_robot_command(cx, cy, bw, bh, FRAME_WIDTH)
            print(f"[DETECTED] cx={cx:4d}  bh={bh:4d}px  -> {command}")
            draw_overlay(frame, cx, cy, bw, bh, command)
        else:
            print("[NO TARGET] STOP")
            cv2.putText(frame, "SEARCHING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("QuadBots - Object Following", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
