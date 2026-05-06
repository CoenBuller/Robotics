import cv2
import numpy as np
import time
from threading import Thread

# import picar_4wd as fc   # <-- UNCOMMENT when using motors

# Number of consecutive frames where detection fails before triggering a
# global re-acquisition search instead of trusting the Kalman prediction.
LOST_FRAME_LIMIT = 15

class CameraStream:
    def __init__(self, cam):
        self.stream = cam
        self.success, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.success, self.frame = self.stream.read()

    def read(self):
        return self.success, self.frame

    def stop(self):
        self.stopped = True


def init_kalman(cx0, cy0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePost           = np.array([[cx0],[cy0],[0],[0]], np.float32)
    return kf


class MaskObjectDetector:
    def __init__(self, template_mask, hv_ranges, template_area):
        self.template      = template_mask
        self.template_area = template_area
        self.hv_ranges     = hv_ranges

    @staticmethod
    def auto_calibrate(frame, contour, std_multiplier=2.0):
        """Compute HS ranges from the pixels inside a contour on `frame`."""
        contour = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # FIX: pass mask so zero background pixels are excluded from the stats.
        # Without this, meanStdDev averages over the whole bounding rectangle,
        # pulling the mean toward 0 and making every range wrong.
        mean, stddev = cv2.meanStdDev(frame, mask=mask)
        mean   = mean.flatten()
        stddev = stddev.flatten()

        lower = np.clip(mean - stddev * std_multiplier, 0, 255).astype(np.uint8)
        upper = np.clip(mean + stddev * std_multiplier, 0, 255).astype(np.uint8)
        return [(lower, upper)]

    def _color_mask(self, frame):
        mask = None
        for lo, hi in self.hv_ranges:
            m = cv2.inRange(frame, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        kernel = np.ones((5, 5), np.uint8)
        # MORPH_CLOSE fills small holes inside the object before MORPH_OPEN
        # removes isolated noise — better than OPEN alone when the object has
        # texture gaps (e.g. highlights on a shiny surface).
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        return cv2.bitwise_not(mask)

    def detect(self, frame, prev_bbox=None, global_search=False):
        """
        Returns (bbox, mask, match_score, contour_in_frame_coords).
        contour_in_frame_coords is None when no reliable contour is found.
        """
        current_mask = self._color_mask(frame)
        res = cv2.matchTemplate(current_mask, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        tx, ty = max_loc
        th, tw = self.template.shape[:2]

        if global_search or prev_bbox is None:
            # Search the whole frame — used during re-acquisition
            roi_y1, roi_x1 = 0, 0
            roi_y2, roi_x2 = current_mask.shape[:2]
        else:
            margin = max(prev_bbox[2], prev_bbox[3]) + 10
            roi_y1 = max(0, ty - margin)
            roi_y2 = min(current_mask.shape[0], ty + th + margin)
            roi_x1 = max(0, tx - margin)
            roi_x2 = min(current_mask.shape[1], tx + tw + margin)

        roi = current_mask[roi_y1:roi_y2, roi_x1:roi_x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > 100:
                x, y, w, h = cv2.boundingRect(largest_cnt)
                # FIX: offset contour back to full-frame coordinates before
                # returning it. Previously the raw ROI-local contour was passed
                # to auto_calibrate with the full frame, so color stats were
                # sampled from the wrong location on every update.
                frame_contour = largest_cnt + np.array([[[roi_x1, roi_y1]]])
                score = cv2.contourArea(largest_cnt) * max_val
                return (x + roi_x1, y + roi_y1, w, h), current_mask, score, frame_contour

        return (tx, ty, tw, th), current_mask, 0.0, None


def TrackerDetector(cam, template, initial_bbox, drive_motors=False):
    stream = CameraStream(cam).start()
    time.sleep(1.0)
    ret, frame = stream.read()
    if not ret:
        return

    SHIFT_THRESH = 100

    # Normalise the template contour to the shape/dtype OpenCV requires everywhere
    template = np.array(template, dtype=np.int32).reshape(-1, 1, 2)

    hs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., :3]
    H, W = frame.shape[:2]

    # FIX: use std_multiplier=1.5 instead of 0.5. 0.5 is so tight that almost
    # no colour variation is accepted, causing the mask to collapse immediately.
    hv_ranges = MaskObjectDetector.auto_calibrate(hs_frame, template, std_multiplier=1.5)

    x, y, w, h = initial_bbox
    tmp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tmp_mask, [template], -1, 255, thickness=cv2.FILLED)
    detector = MaskObjectDetector(tmp_mask, hv_ranges, cv2.contourArea(template))

    est_cx, est_cy = float(x + w / 2), float(y + h / 2)
    ref_area = w * h
    kf = init_kalman(est_cx, est_cy)

    last_action = None
    prev_bbox   = initial_bbox
    lost_frames = 0   # consecutive frames without a reliable detection

    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                continue

            hs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., :3]

            pred = kf.predict().flatten()
            pred_cx, pred_cy = float(pred[0]), float(pred[1])

            # After LOST_FRAME_LIMIT frames, drop the ROI restriction and search
            # globally so the tracker can re-acquire the object.
            global_search = lost_frames >= LOST_FRAME_LIMIT
            bbox_det, mask, match_score, contour = detector.detect(
                hs_frame, prev_bbox=prev_bbox, global_search=global_search
            )

            jump = (
                np.abs(bbox_det[0] - pred_cx) > SHIFT_THRESH or
                np.abs(bbox_det[1] - pred_cy) > SHIFT_THRESH or
                contour is None
            )

            if jump:
                bbox_det    = (pred_cx, pred_cy, prev_bbox[2], prev_bbox[3])
                lost_frames += 1
            else:
                lost_frames = 0   # good detection — color ranges stay fixed

            prev_bbox = bbox_det

            x, y, w, h = (int(v) for v in bbox_det)
            est      = kf.correct(np.array([[np.float32(x + w/2)],
                                            [np.float32(y + h/2)]])).flatten()
            est_cx, est_cy = float(est[0]), float(est[1])

            area_ratio = (w * h) / ref_area
            err_x      = est_cx - W / 2

            if   area_ratio > 1.8: action = ("BACKWARD", 20)
            elif area_ratio < 0.6: action = ("FORWARD",  25)
            elif abs(err_x)  > 50: action = ("LEFT", 1) if err_x < 0 else ("RIGHT", 1)
            else:                  action = ("IDLE", 0)

            if action != last_action and drive_motors:
                if   action[0] == "IDLE":     fc.stop()
                elif action[0] == "FORWARD":  fc.forward(action[1])
                elif action[0] == "BACKWARD": fc.backward(action[1])
                elif action[0] == "LEFT":     fc.turn_left(action[1])
                elif action[0] == "RIGHT":    fc.turn_right(action[1])
                last_action = action

            # --- UI ---
            status = "LOST (searching...)" if global_search else action[0]
            color  = (0, 0, 255) if global_search else (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.rectangle(frame,
                          (int(est_cx-w/2), int(est_cy-h/2)),
                          (int(est_cx+w/2), int(est_cy+h/2)), color, 3)
            cv2.putText(frame, f"Ratio:{area_ratio:.2f}  {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Automated RGB Tracker", frame)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.rectangle(mask_bgr,
                          (int(est_cx-w/2), int(est_cy-h/2)),
                          (int(est_cx+w/2), int(est_cy+h/2)), color, 3)
            cv2.imshow("Color Mask", mask_bgr)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()