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

    # Motor / control constants
    POWER_TURN       = 4
    POWER_FORWARD    = 4
    THRESHOLD_X      = 60
    THRESHOLD_Y      = 30
    RAMP_DURATION    = 2
    RAMP_START_POWER = 1

    TRACK_PAUSE_AFTER_ACTION = 0.3

    last_action     = None
    last_sent       = None
    action_start_t  = time.time()

    # Initialise last-known Kalman box (updated every frame on success)
    last_x = int(ix)
    last_y = int(iy)
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

                cx, cy = x + w / 2, y + h / 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)           # CSRT (green)

                # Store Kalman position as reference for next frame's jump guard
                last_x, last_y, last_w, last_h = x, y, w, h

                err_x = cx - init_cx
                err_y = init_cy - cy   # positive when object is above frame centre

                target_action = getAction(err_x, err_y, THRESHOLD_X, THRESHOLD_Y, POWER_FORWARD, POWER_TURN)

                # Power ramping, smooth direction changes
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
                cv2.rectangle(frame, (last_x, last_y ), (last_x + last_w, last_y + last_y), (0, 200, 200), 2)
                cv2.putText(frame, "settling", (last_x, last_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            else:
                # Genuinely lost
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