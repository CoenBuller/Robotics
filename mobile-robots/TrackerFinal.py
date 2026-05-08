import cv2
import numpy as np
# import picar_4wd as fc
from threading import Thread
import time

def getAction(err_x, err_y, threshold_x, threshold_y, power_forward, power_turn):
    
    abs_x, abs_y = abs(err_x), abs(err_y)
    
    if abs_x < threshold_x and abs_y < threshold_y:  # Dont do anything if errors are not big enough
        return ("STOP", 0)

    # If the error on y is negative, move backwards, else move forward
    if err_y < 0:
        movey = ("BACKWARDS", power_forward)
    else:
        movey = ("FORWARD", power_forward)


    if abs_x < threshold_x: # If horizontally the object is close enough to center, move forward 
        return movey
    
    # If the error on x is negative, we have to move left, right otherwise
    if err_x < 0:
        movex = ("TURN_LEFT", power_turn)
    else: 
        movex = ("TURN_RIGHT", power_turn)
    

    # If the error on the horizontal position is larger than the veritcal position we turn. Move forward otherwise
    if abs(err_x) > abs(err_y): 
        return movex
    else:
        return movey


def init_kalman(cx0, cy0):
    """4-state (x, y, dx, dy) / 2-measurement Kalman filter.
    Process noise raised so abrupt robot-motion-induced jumps don't blow up the
    velocity estimate."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx0], [cy0], [0], [0]], np.float32)
    return kf


# Threaded camera reader
class CameraStream:
    def __init__(self, cam):
        self.stream = cam
        self.success, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.success, self.frame = self.stream.read()

    def read(self):
        return self.success, self.frame

    def stop(self):
        self.stopped = True


# Main tracker
def Tracker(cam, initial_bbox, drive_motors=False, scale=0.5):
    """
    Use CSRT tracker to track the object confined in the initial bbox. To reduce the computational 
    load on the raspberry pi we scale the image size down by a factor of 0.

    To make the tracking of the filter more robust to sudden jumps, we'll also use a kalman filter
    to smoothen out the motion of the bounding box. 

    Features:
      * Kalman process noise raised — better at tolerating sudden jumps.
      * Track pause: ignore tracker output for TRACK_PAUSE_AFTER_ACTION
        seconds after every direction change, so the robot's own motion
        doesn't confuse the tracker.
      * Motor power capped low for safety.
      * Power ramping on direction changes — robot doesn't lurch.
    """

    stream = CameraStream(cam).start()
    time.sleep(1.0)

    ret, frame = stream.read()
    if not ret or frame is None or frame.size == 0:
        print("Error: could not read initial frame for tracker.")
        return

    H, W = frame.shape[:2]
    Hs, Ws = int(H * scale), int(W * scale)
    inv = 1.0 / scale

    ix, iy, iw, ih = initial_bbox
    init_cx, init_cy = int(ix + iw/2), int(iy + ih/2)
    init_bbox_small = (
        int(ix * scale),
        int(iy * scale),
        int(iw * scale),
        int(ih * scale),
    )

    small     = cv2.resize(frame, (Ws, Hs))
    small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) # Use HSV instead of RGB -> More robust to varying lighting

    # Initialize tracker
    tracker   = cv2.TrackerCSRT_create()
    tracker.init(small_hsv, init_bbox_small)

    # Initialize Kalman tracker
    kf = init_kalman(ix + iw / 2, iy + ih / 2)

    # Motor / control constants
    POWER_TURN    = 0.5
    POWER_FORWARD = 0.5
    THRESHOLD_X   = 60
    THRESHOLD_Y   = 30
    RAMP_DURATION    = 0.5
    RAMP_START_POWER = 1

    TRACK_PAUSE_AFTER_ACTION = 0.5

    last_action     = None
    last_sent       = None
    action_start_t  = time.time()

    # Initialise last-known Kalman box (updated every frame on success)
    last_kx = int(ix)
    last_ky = int(iy)
    last_w  = int(iw)
    last_h  = int(ih)

    print(f"Tracker on {Ws}x{Hs} (scale={scale}), display at {W}x{H}. ESC to stop.")

    fps_t0 = time.time()
    fps_n  = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None or frame.size == 0:
                continue

            # Kalman prediction (always runs, even when CSRT fails)
            prediction = kf.predict().flatten()
            pred_cx = float(prediction[0])
            pred_cy = float(prediction[1])

            # Compute settling flag before tracker.update so both branches can use it
            settling = (time.time() - action_start_t) < TRACK_PAUSE_AFTER_ACTION

            small     = cv2.resize(frame, (Ws, Hs))
            small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            # CSRT update
            success, bbox_small = tracker.update(small_hsv)

            if success:
                sx, sy, sw, sh = bbox_small
                x = int(sx * inv)
                y = int(sy * inv)
                w = int(sw * inv)
                h = int(sh * inv)

                # Jump guard: compare new CSRT centre against last *Kalman* centre
                new_cx = x + (sw * inv) / 2
                new_cy = y + (sh * inv) / 2
                last_kcx = last_kx + last_w / 2
                last_kcy = last_ky + last_h / 2
                jump = ((new_cx - last_kcx) ** 2 + (new_cy - last_kcy) ** 2) ** 0.5
                if jump > 150:   # bbox moved implausibly far — treat as lost
                    success = False

            if success:
                cx, cy = x + w / 2, y + h / 2
                meas      = np.array([[np.float32(cx)], [np.float32(cy)]])
                estimated = kf.correct(meas).flatten()
                est_cx    = float(estimated[0])
                est_cy    = float(estimated[1])

                # Kalman-corrected bounding box — this is the box we track
                kx = int(est_cx - w / 2)
                ky = int(est_cy - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)           # CSRT (green)
                cv2.rectangle(frame, (kx, ky), (kx + w, ky + h), (255, 100, 0), 1)     # Kalman (blue)

                # Store Kalman position as reference for next frame's jump guard
                last_kx, last_ky, last_w, last_h = kx, ky, w, h

                err_x = est_cx - init_cx
                err_y = init_cy - est_cy   # positive when object is above frame centre

                target_action = getAction(err_x, err_y, THRESHOLD_X, THRESHOLD_Y, POWER_FORWARD, POWER_TURN)

                # Power ramping — smooth direction changes
                if last_action is None or last_action[0] != target_action[0]:
                    action_start_t = time.time()
                    last_action = target_action

                target_name, target_power = target_action
                t_in_action = time.time() - action_start_t
                if t_in_action < RAMP_DURATION:
                    ramp_t = t_in_action / RAMP_DURATION
                    ramped_power = int(RAMP_START_POWER + ramp_t * (target_power - RAMP_START_POWER))
                    ramped_power = max(RAMP_START_POWER, min(target_power, ramped_power))
                else:
                    ramped_power = target_power

                action_to_send = (target_name, ramped_power)

                if action_to_send != last_sent:
                    name, power = action_to_send
                    print(f"{name:9s} power={power}  err_x={err_x:+6.1f}")
                    if drive_motors:
                        if   name == "TURN_LEFT" : fc.turn_left(power)
                        elif name == "TURN_RIGHT": fc.turn_right(power)
                        elif name == "FORWARD"   : fc.forward(power)
                        elif name == "BACKWARDS" : fc.backward(power)
                        else                     : fc.stop()
                    last_sent = action_to_send

            elif settling:
                # Robot is still moving from a recent action, don't update motors
                w_, h_ = last_w, last_h
                kx = int(pred_cx - w_ / 2)
                ky = int(pred_cy - h_ / 2)
                cv2.rectangle(frame, (kx, ky), (kx + w_, ky + h_), (0, 200, 200), 2)
                cv2.putText(frame, "settling", (kx, ky - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            else:
                # Genuinely lost, fall back to Kalman prediction and stop motors
                w_, h_ = last_w, last_h
                kx = int(pred_cx - w_ / 2)
                ky = int(pred_cy - h_ / 2)
                cv2.rectangle(frame, (kx, ky), (kx + w_, ky + h_), (0, 0, 255), 2)
                cv2.putText(frame, "Lost (Kalman)", (kx, ky - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if last_sent != ("STOP", 0):
                    print("LOST — stopping")
                    if drive_motors:
                        fc.stop()
                    last_sent   = ("STOP", 0)
                    last_action = ("STOP", 0)

            fps_n += 1
            if fps_n >= 30:
                fps   = fps_n / (time.time() - fps_t0)
                print(f"FPS: {fps:.1f}")
                fps_n  = 0
                fps_t0 = time.time()

            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        stream.stop()
        if drive_motors:
            fc.stop()
        cv2.destroyAllWindows()