#!/usr/bin/env python3

import sys
sys.path.append('/home/pi/robotics/colours')

from color_detect import color_detect,color_dict,kernel_5
import picar_4wd as fc
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# â€â€ Config â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€
POWER             = 15
SEARCH_POWER      = 10
EVAL_DURATION     = 60
FRAME_W, FRAME_H  = 640, 480
TARGET_AREA_RATIO = 0.06      # tune with calibration
AREA_DEADBAND     = 0.010
STEER_THRESHOLD   = 0.20

ROI_X1 = FRAME_W // 4        # 160
ROI_X2 = 3 * FRAME_W // 4    # 480
ROI_Y1 = FRAME_H // 4        # 120
ROI_Y2 = 3 * FRAME_H // 4    # 360

ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_100,
]

# â€â€ PID â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€â€
class PID:
    def __init__(self, kp, ki, kd, limit=100):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self._i = self._prev_e = 0.0
        self._prev_t = time.time()

    def compute(self, error):
        now = time.time()
        dt  = max(now - self._prev_t, 1e-4)
        self._i    += error * dt
        d           = (error - self._prev_e) / dt
        out         = self.kp*error + self.ki*self._i + self.kd*d
        self._prev_e, self._prev_t = error, now
        return float(np.clip(out, -self.limit, self.limit))

# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# STRATEGY 1: ArUco
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
def aruco_try_detect(img):
    """Try all ArUco dicts, return (aruco_dict, aruco_params, dict_id) or None."""
    for dict_id in ARUCO_DICTS:
        aruco_dict   = cv2.aruco.Dictionary_get(dict_id)
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, aruco_dict, parameters=aruco_params)
        if ids is not None and len(ids) > 0:
            return aruco_dict, aruco_params, dict_id
    return None

def aruco_detect(img, strategy_data):
    aruco_dict, aruco_params = strategy_data
    corners, ids, _ = cv2.aruco.detectMarkers(
        img, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        return None, None, img

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    all_corners = np.concatenate([c[0] for c in corners], axis=0)
    x, y, w, h  = cv2.boundingRect(all_corners.astype(np.int32))

    cx_px      = x + w / 2
    area_ratio = (w * h) / (FRAME_W * FRAME_H)
    cx_norm    = (cx_px - FRAME_W / 2) / (FRAME_W / 2)

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(img, f"ArUco cx={cx_norm:+.2f} area={area_ratio:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return cx_norm, area_ratio, img

def color_detection(img,strategy_data):
    color_name=strategy_data
    annotated_img,mask,morphed=color_detect(img,color_name)
    color_tuple=cv2.findContours(morphed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=color_tuple[1] if len(color_tuple)==3 else color_tuple[0]

    if not contours:
        return None,None,annotated_img
    best = max(contours,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(best)

    if w<8 or h<8:
        return None,None,annotated_img
    x,y,w,h=x*4,y*4,w*4,h*4

    cx_px=x+w/2
    area_ratio=(w*h)/(FRAME_W*FRAME_H)
    cx_norm=(cx_px - FRAME_W/2)/(FRAME_W/2)
    return cx_norm,area_ratio,annotated_img

def shape_learn(bg_frames, obj_frames):
    if len(bg_frames) == 0:
        return None

    bg_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                         for f in bg_frames], axis=0)
    bg_model = np.median(bg_stack, axis=0).astype(np.uint8)
    return bg_model

def color_learn(frames):
    best_color = None
    best_score = 0
    sample_frames = frames[-10:] if len(frames) >= 10 else frames

    for color_name in color_dict:
        if color_name == 'red_2':
            continue
        score = 0
        for img in sample_frames:
            roi   = img[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            small = cv2.resize(roi, (160, 120), interpolation=cv2.INTER_LINEAR)
            hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            lo    = np.array([min(color_dict[color_name]), 60, 60])
            hi    = np.array([max(color_dict[color_name]), 255, 255])
            mask  = cv2.inRange(hsv, lo, hi)
            if color_name == 'orange':
                mask2 = cv2.inRange(hsv,
                    np.array([color_dict['red_2'][0], 0,   0  ]),
                    np.array([color_dict['red_2'][1], 255, 255]))
                mask = cv2.bitwise_or(mask, mask2)
            score += cv2.countNonZero(mask)
        if score > best_score:
            best_score = score
            best_color = color_name

    if best_score < 500:
        print("  No strong color found")
        return None

    print(f"  Dominant color: {best_color} (score={best_score})")
    return best_color

def shape_detect(img, strategy_data):
    bg_model = strategy_data
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_model)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:
        return None, None, img
    cx_px      = x + w / 2
    area_ratio = (w * h) / (FRAME_W * FRAME_H)
    cx_norm    = (cx_px - FRAME_W / 2) / (FRAME_W / 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 100, 0), 2)
    cv2.putText(img, f"Shape cx={cx_norm:+.2f} area={area_ratio:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    return cx_norm, area_ratio, img

def detect_object(img, strategy_name, strategy_data):
    if strategy_name == "aruco":
        return aruco_detect(img, strategy_data)
    elif strategy_name == "color":
        return color_detection(img, strategy_data)
    elif strategy_name == "shape":
        return shape_detect(img, strategy_data)
    return None, None, img


def observation_phase(cap):

    print("Observation phase started...")
    frame_count = 0
    MAX_FRAMES = 1000  # ~2-3 seconds depending on FPS

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()  
        if not ret:
            break

        fgmask = frame

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find contours (moving objects)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fgmask, contours, -1, (0, 255), 3)
        cv2.imshow('Contours', fgmask)

        # cv2.imshow("Observation", frame)
        # cv2.imshow("FG Mask", fgmask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    

    # cap.release()
    # cv2.destroyAllWindows()
# 
# EVALUATION PHASE
# 
def evaluation_phase(camera):
    """
    60 second evaluation:
    - First 10s: capture background (no object yet)
    - After 10s: object is placed â analyze it
    - Lock onto best strategy (ArUco > Color > Shape)
    """
    print("\nEVALUATION PHASE - analyzing environment...")
    print(f" Duration: {EVAL_DURATION}s")
    print("First 10s: keep object OUT of view (background learning)")
    print("After 10s: place object in front of robot\n")

    bg_frames    = []
    obj_frames   = []
    aruco_counts = {}
    start        = time.time()
    strategy_name = None
    strategy_data = None
    confirmed     = False
    while time.time() - start < EVAL_DURATION:
        img     = camera.capture_array()
        elapsed = time.time() - start
        remaining = EVAL_DURATION - elapsed
        phase = "BACKGROUND" if elapsed < 10 else "OBJECT ANALYSIS"
        cv2.putText(img, f"{phase}  {remaining:.0f}s remaining",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        # Draw ROI box
        cv2.rectangle(img, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (200, 200, 0), 1)
        if elapsed < 10:
            # Background learning phase
            bg_frames.append(img.copy())
            cv2.putText(img, "Keep object OUT of view",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            obj_frames.append(img.copy())
            if not confirmed:
                #  Try ArUco first 
                result = aruco_try_detect(img)
                if result:
                    aruco_dict, aruco_params, dict_id = result
                    aruco_counts[dict_id] = aruco_counts.get(dict_id, 0) + 1
                    if aruco_counts[dict_id] >= 10:
                        strategy_name = "aruco"
                        strategy_data = (aruco_dict, aruco_params)
                        confirmed     = True
                        print(f"\n  Strategy LOCKED: ArUco (dict={dict_id})")
                #  Try Color after 15s if ArUco not found 
                elif elapsed > 15 and len(obj_frames) >= 20:
                    color_range = color_learn(obj_frames[-20:])
                    if color_range:
                        # Verify it actually detects something
                        cx, ar, _ = color_detect(img, color_range)
                        if cx is not None:
                            strategy_name = "color"
                            strategy_data = color_range
                            confirmed     = True
                            print(f"\n  Strategy LOCKED: Color HSV {color_range}")
                #  Fallback to Shape after 25s 
                elif elapsed > 25 and len(bg_frames) >= 10:
                    bg_model = shape_learn(bg_frames, obj_frames)
                    if bg_model is not None:
                        strategy_name = "shape"
                        strategy_data = bg_model
                        confirmed     = True
                        print(f"\n  Strategy LOCKED: Background subtraction")
            if confirmed:
                # Show live detection during remaining eval time
                cx, ar, img = detect_object(img, strategy_name, strategy_data)
                cv2.putText(img, f"LOCKED: {strategy_name.upper()}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
        cv2.imshow("Evaluation", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if not confirmed:
        print("\n  WARNING: No strategy confirmed â€ using shape as last resort")
        bg_model = shape_learn(bg_frames, obj_frames)
        if bg_model is not None:
            strategy_name = "shape"
            strategy_data = bg_model
        else:
            print("  FATAL: Could not lock any strategy")
    print(f"\n  Final strategy: {strategy_name}")
    return strategy_name, strategy_data

# FOLLOWING PHASE

def following_phase(camera, strategy_name, strategy_data):
    print(f"\nFOLLOWING PHASE - strategy: {strategy_name}")
    print("   Press ESC to stop\n")
    dist_pid = PID(kp=80, ki=0.5, kd=5.0)
    while True:
        img = camera.capture_array()
        cx_norm, area_ratio, img = detect_object(img, strategy_name, strategy_data)
        if cx_norm is not None:
            area_error = area_ratio - TARGET_AREA_RATIO
            if abs(cx_norm) > STEER_THRESHOLD:
                if cx_norm > 0:
                    fc.turn_right(POWER)
                    action = f"TURN RIGHT  cx={cx_norm:+.2f}"
                else:
                    fc.turn_left(POWER)
                    action = f"TURN LEFT cx={cx_norm:+.2f}"
            elif area_error > AREA_DEADBAND:
                spd = int(np.clip(abs(dist_pid.compute(-area_error)), 20, POWER+20))
                fc.backward(spd)
                action = f"BACKWARD area={area_ratio:.3f}"
            elif area_error < -AREA_DEADBAND:
                spd = int(np.clip(abs(dist_pid.compute(-area_error)), 20, POWER+20))
                fc.forward(spd)
                action = f"FORWARD area={area_ratio:.3f}"
            else:
                fc.stop()
                action = f"HOLD area={area_ratio:.3f}"
        else:
            fc.stop()
            action = "NOT FOUND - waiting"
        cv2.imshow("Following", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# MAIN
def setup_camera():
    camera = Picamera2()
    camera.preview_configuration.main.size   = (FRAME_W, FRAME_H)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()
    time.sleep(1)
    return camera

def main():
    print("Unknown Object Follower - Mobile Robot Challenge II")
    print("=" * 50)
    camera = setup_camera()
    try:
        observation_phase(camera)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        fc.stop()
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()