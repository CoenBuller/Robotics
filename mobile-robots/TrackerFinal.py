import cv2
import numpy as np
# import picar_4wd as fc
from threading import Thread
import time


# Helpers
def GetObjectDensity(frame, bbox):
    """Foreground pixel count inside bbox using HSV saturation + Otsu."""
    x, y, w, h = [int(v) for v in bbox]
    roi = frame[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.countNonZero(mask)


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
    ref_area = iw * ih

    small     = cv2.resize(frame, (Ws, Hs))
    small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) # Use HSV instead of RGB -> More robust to varying lighting

    # Initialize tracker
    tracker   = cv2.TrackerCSRT_create()
    tracker.init(small_hsv, init_bbox_small)

    # Initialize Kalman tracker
    kf = init_kalman(ix + iw / 2, iy + ih / 2)

    init_pixel_dens = GetObjectDensity(frame, initial_bbox)
    SHRUNK_PATIENCE = 10
    shrunk_count    = SHRUNK_PATIENCE

    # Motor / control constants
    POWER_TURN    = 0.5
    POWER_FORWARD = 0.5
    DEADBAND    = 60

    RAMP_DURATION    = 0.5
    RAMP_START_POWER = 1

    TRACK_PAUSE_AFTER_ACTION = 0.5

    last_action     = None
    last_sent       = None
    action_start_t  = time.time()
    proximity_stop  = False

    last_x = ix
    last_y = iy
    last_w = iw
    last_h = ih

    print(f"Tracker on {Ws}x{Hs} (scale={scale}), display at {W}x{H}. ESC to stop.")

    fps_t0 = time.time()
    fps_n  = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None or frame.size == 0:
                continue

            small     = cv2.resize(frame, (Ws, Hs))
            small_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            # Make prediction with Kalman filter which will be used as reference
            prediction = kf.predict()
            pred_cx = float(prediction[0])
            pred_cy = float(prediction[1])

            # Ignore tracker for a short time after action so the robot does not stutter
            t_since_action = time.time() - action_start_t
            settling = t_since_action < TRACK_PAUSE_AFTER_ACTION

            # Get CSRT tracker prediction
            success, bbox_small = tracker.update(small_hsv)

            if success:
                sx, sy, sw, sh = bbox_small
                # Scale bounding box sizes back to real image size
                x = int(sx * inv)
                y = int(sy * inv)
                w = int(sw * inv)
                h = int(sh * inv)
                new_cx=x+(sw*inv)/2
                new_cy=y+(sh*inv)/2

                last_cx=last_x+last_w/2
                last_cy=last_y+last_h/2

                jump=((new_cx-last_cx)**2 + (new_cy-last_cy)**2)**0.5
                if jump > 150: # Make shure the bbox does not move too much (e.g. false prediction)
                    success=False


                cx, cy = x + w / 2, y + h / 2
                meas      = np.array([[np.float32(cx)], [np.float32(cy)]])
                estimated = kf.correct(meas).flatten()
                est_cx    = float(estimated[0])
                est_cy    = float(estimated[1])

                kx = int(est_cx - w / 2)
                ky = int(est_cy - h / 2)
                kalman_box = (kx, ky, w, h)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (kx, ky), (kx + w, ky + h), (255, 100, 0), 1)

                last_x, last_y, last_w, last_h = x, y, w, h    

                err_x = est_cx - W / 2
                err_y = est_cy - init_cy


                if   err_x < -DEADBAND: target_action = ("TURN_LEFT",  POWER_TURN)
                elif err_x >  DEADBAND: target_action = ("TURN_RIGHT", POWER_TURN)
                elif err_y > DEADBAND:  target_action = ("FORWARD",    POWER_FORWARD)

                if last_action is None or last_action[0] != target_action[0]:
                    action_start_t = time.time()
                    last_action = target_action

                target_name, target_power = target_action
                t_in_action = time.time() - action_start_t
                if t_in_action < RAMP_DURATION:
                    ramp_t = t_in_action / RAMP_DURATION
                    ramped_power = int(RAMP_START_POWER +
                                        ramp_t * (target_power - RAMP_START_POWER))
                    ramped_power = max(RAMP_START_POWER,
                                        min(target_power, ramped_power))
                else:
                    ramped_power = target_power

                action_to_send = (target_name, ramped_power)

                if action_to_send != last_sent:
                    name, power = action_to_send
                    print(f"{name:9s} power={power}  err_x={err_x:+6.1f}")
                    if drive_motors:
                        if   name == "TURN_LEFT":  fc.turn_left(power)
                        elif name == "TURN_RIGHT": fc.turn_right(power)
                        else:                       fc.forward(power)
                    last_sent = action_to_send
            if success and last_x is not None:
                sx, sy, sw, sh = bbox_small
                new_x = int(sx * inv)
                new_y = int(sy * inv)
                new_cx = new_x + (sw * inv) / 2
                new_cy = new_y + (sh * inv) / 2
                last_cx = last_x + last_w / 2
                last_cy = last_y + last_h / 2
                jump = ((new_cx - last_cx)**2 + (new_cy - last_cy)**2) ** 0.5
                if jump > 100:
                    success = False
            elif settling:
                w_, h_ = last_w, last_h
                kx = int(pred_cx - w_ / 2)
                ky = int(pred_cy - h_ / 2)
                cv2.rectangle(frame, (kx, ky), (kx + w_, ky + h_), (0, 200, 200), 2)
                cv2.putText(frame, "settling", (kx, ky - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            else:
                w_, h_ = iw, ih
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

                proximity_stop = False
                shrunk_count   = SHRUNK_PATIENCE

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
