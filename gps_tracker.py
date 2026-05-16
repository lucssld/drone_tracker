'''
Hybrid YOLO + Optical Flow Tracker
WITH MAVLINK TARGET GPS ESTIMATION

Optimized for Raspberry Pi

Features:
- Sparse YOLO detections
- Optical flow tracking
- ROI-only detections
- Manual target selection
- MAVLink telemetry
- Live target GPS estimation

Controls:
W A S D = Move targeting box
T = Reset tracking
Q = Quit
'''

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from pymavlink import mavutil

import threading
import math

print('\n-----SETUP-----\n')

directory_path = Path('.')

# =========================================================
# MAVLINK CONFIG
# =========================================================

SERIAL_PORT = "/dev/serial0"
BAUD = 115200

CAMERA_FOV_H = 78
CAMERA_FOV_V = 64

TARGET_WIDTHS = {
    "person": 0.45,
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.8
}

# =========================================================
# TELEMETRY STATE
# =========================================================

drone_lat = None
drone_lon = None
drone_alt = None
drone_heading = None

# =========================================================
# CONNECT MAVLINK
# =========================================================

print("Connecting MAVLink...")

master = mavutil.mavlink_connection(
    SERIAL_PORT,
    baud=BAUD
)

master.wait_heartbeat()

print("MAVLink Connected")

# =========================================================
# TELEMETRY THREAD
# =========================================================

def telemetry_loop():

    global drone_lat
    global drone_lon
    global drone_alt
    global drone_heading

    while True:

        msg = master.recv_match(
            type='GLOBAL_POSITION_INT',
            blocking=True
        )

        if not msg:
            continue

        drone_lat = msg.lat / 1e7
        drone_lon = msg.lon / 1e7

        drone_alt = (
            msg.relative_alt / 1000.0
        )

        drone_heading = (
            msg.hdg / 100.0
        )

threading.Thread(
    target=telemetry_loop,
    daemon=True
).start()

# =========================================================
# GPS OFFSET
# =========================================================

def offset_gps(lat, lon, north_m, east_m):

    dlat = north_m / 111320.0

    dlon = east_m / (
        111320.0 *
        math.cos(math.radians(lat))
    )

    return (
        lat + dlat,
        lon + dlon
    )

# =========================================================
# TARGET ESTIMATION
# =========================================================

def estimate_target_position(
    bbox,
    class_name
):

    global drone_lat
    global drone_lon
    global drone_alt
    global drone_heading

    if None in (
        drone_lat,
        drone_lon,
        drone_alt,
        drone_heading
    ):
        return None

    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1

    cx = (x1 + x2) / 2

    # -----------------------------------------------------
    # Horizontal screen offset
    # -----------------------------------------------------

    norm_x = (
        (cx - width / 2)
        / (width / 2)
    )

    yaw_offset = (
        norm_x *
        (CAMERA_FOV_H / 2)
    )

    # -----------------------------------------------------
    # Distance estimate
    # -----------------------------------------------------

    real_width = TARGET_WIDTHS.get(
        class_name,
        1.0
    )

    focal_px = (
        width /
        (
            2 *
            math.tan(
                math.radians(
                    CAMERA_FOV_H / 2
                )
            )
        )
    )

    est_distance = (
        real_width *
        focal_px
    ) / max(bbox_w, 1)

    # -----------------------------------------------------
    # Bearing
    # -----------------------------------------------------

    bearing = (
        drone_heading +
        yaw_offset
    ) % 360

    bearing_rad = math.radians(
        bearing
    )

    north = (
        est_distance *
        math.cos(bearing_rad)
    )

    east = (
        est_distance *
        math.sin(bearing_rad)
    )

    target_lat, target_lon = offset_gps(
        drone_lat,
        drone_lon,
        north,
        east
    )

    return {
        "lat": target_lat,
        "lon": target_lon,
        "distance": est_distance,
        "bearing": bearing
    }

# =========================================================
# Model Selection
# =========================================================

while True:

    model_list = list(
        directory_path.glob('*.pt')
    )

    print('\nAvailable YOLO models:')

    for i, model_file in enumerate(
        model_list,
        start=1
    ):
        print(
            f'{i} - {model_file.name}'
        )

    model_selection = int(
        input('\nSelect model number: ')
    )

    if 1 <= model_selection <= len(model_list):

        selected_model = model_list[
            model_selection - 1
        ]

        break

    print("Invalid selection.\n")

model = YOLO(selected_model)

# =========================================================
# Video Source Selection
# =========================================================

selection = False

while not selection:

    cap_method = input(
        'Select video source -> '
        '[1] Webcam | [2] Video File: '
    )

    if cap_method == '1':

        cap = cv2.VideoCapture(0)

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            360
        )

        if not cap.isOpened():

            print(
                "Could not open webcam."
            )

            exit()

        selection = True

    elif cap_method == '2':

        while True:

            vid_list = (
                list(
                    directory_path.glob('*.mov')
                ) +
                list(
                    directory_path.glob('*.mp4')
                ) +
                list(
                    directory_path.glob('*.avi')
                )
            )

            print(
                '\nAvailable video files:'
            )

            for i, vid_file in enumerate(
                vid_list,
                start=1
            ):
                print(
                    f'{i} - {vid_file.name}'
                )

            vid_selection = int(
                input(
                    '\nSelect video number: '
                )
            )

            if (
                1 <= vid_selection <=
                len(vid_list)
            ):

                selected_video = vid_list[
                    vid_selection - 1
                ]

                cap = cv2.VideoCapture(
                    str(selected_video)
                )

                selection = True

                break

            print("Invalid selection.\n")

    else:
        print("Invalid input.\n")

# =========================================================
# Video Properties
# =========================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# =========================================================
# Target Box Size
# =========================================================

user = input(
    "\nEnter targeter size | "
    "[1] Small (50x50) | "
    "[2] Medium (100x100) | "
    "[3] Large (150x150): "
)

if user == '1':

    box_w, box_h = 50, 50

elif user == '2':

    box_w, box_h = 100, 100

elif user == '3':

    box_w, box_h = 150, 150

else:

    print(
        "Invalid input. "
        "Defaulting to Medium."
    )

    box_w, box_h = 100, 100

input("Press Enter to start...")

# =========================================================
# Parameters
# =========================================================

MOVE_SPEED = 15

MAX_LOST_FRAMES = 30

IOU_THRESHOLD = 0.3

frame_skip = 10

ROI_PADDING_UNLOCKED = 150

ROI_PADDING_LOCKED = 100

# =========================================================
# Tracking State
# =========================================================

box_x = width // 2 - box_w // 2
box_y = height // 2 - box_h // 2

tracked_box = None
tracked_class = None

locked = False

lost_frames = 0

frame_count = 0

prev_gray = None

track_points = None

target_gps = None

# =========================================================
# Utility Functions
# =========================================================

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(
        0,
        xB - xA
    ) * max(
        0,
        yB - yA
    )

    if interArea == 0:
        return 0.0

    boxAArea = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    boxBArea = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    return interArea / float(
        boxAArea +
        boxBArea -
        interArea
    )

def make_box(x, y, w, h):

    return np.array(
        [x, y, x + w, y + h],
        dtype=np.float32
    )

def center_frame():

    return (
        width // 2 - box_w // 2,
        height // 2 - box_h // 2
    )

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if prev_gray is None:
        prev_gray = gray.copy()

    manual_box = make_box(
        box_x,
        box_y,
        box_w,
        box_h
    )

    # =====================================================
    # OPTICAL FLOW
    # =====================================================

    if (
        locked and
        tracked_box is not None and
        track_points is not None
    ):

        new_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                track_points,
                None,
                winSize=(15, 15),
                maxLevel=2
            )
        )

        if new_points is not None:

            good_new = new_points[
                status == 1
            ]

            good_old = track_points[
                status == 1
            ]

            if len(good_new) > 0:

                movement = (
                    good_new - good_old
                )

                dx, dy = movement.mean(
                    axis=0
                )

                tracked_box[0] += dx
                tracked_box[1] += dy
                tracked_box[2] += dx
                tracked_box[3] += dy

                track_points = (
                    good_new.reshape(
                        -1,
                        1,
                        2
                    )
                )

            else:

                locked = False
                tracked_box = None
                track_points = None

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    if frame_count % frame_skip == 0:

        # -------------------------------------------------
        # ROI
        # -------------------------------------------------

        if locked and tracked_box is not None:

            tx1, ty1, tx2, ty2 = map(
                int,
                tracked_box
            )

            roi_x1 = max(
                0,
                tx1 - ROI_PADDING_LOCKED
            )

            roi_y1 = max(
                0,
                ty1 - ROI_PADDING_LOCKED
            )

            roi_x2 = min(
                width,
                tx2 + ROI_PADDING_LOCKED
            )

            roi_y2 = min(
                height,
                ty2 + ROI_PADDING_LOCKED
            )

        else:

            roi_x1 = max(
                0,
                box_x -
                ROI_PADDING_UNLOCKED
            )

            roi_y1 = max(
                0,
                box_y -
                ROI_PADDING_UNLOCKED
            )

            roi_x2 = min(
                width,
                box_x +
                box_w +
                ROI_PADDING_UNLOCKED
            )

            roi_y2 = min(
                height,
                box_y +
                box_h +
                ROI_PADDING_UNLOCKED
            )

        roi_frame = frame[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        if roi_frame.size != 0:

            results = model(
                roi_frame,
                imgsz=320,
                verbose=False
            )

            boxes = results[0].boxes

            detections = []

            if (
                boxes is not None and
                len(boxes) > 0
            ):

                local_boxes = (
                    boxes.xyxy.cpu().numpy()
                )

                classes = (
                    boxes.cls.cpu().numpy()
                )

                for det, cls_id in zip(
                    local_boxes,
                    classes
                ):

                    det[0] += roi_x1
                    det[1] += roi_y1
                    det[2] += roi_x1
                    det[3] += roi_y1

                    class_name = (
                        model.names[
                            int(cls_id)
                        ]
                    )

                    detections.append(
                        (
                            det,
                            class_name
                        )
                    )

            # ------------------------------------------------
            # REACQUIRE TRACK
            # ------------------------------------------------

            if locked and tracked_box is not None:

                best_iou = 0
                best_det = None
                best_class = None

                for det, class_name in detections:

                    overlap = iou(
                        det,
                        tracked_box
                    )

                    if overlap > best_iou:

                        best_iou = overlap
                        best_det = det
                        best_class = class_name

                if (
                    best_det is not None and
                    best_iou > IOU_THRESHOLD
                ):

                    tracked_box = best_det
                    tracked_class = best_class

                    lost_frames = 0

                    x1, y1, x2, y2 = map(
                        int,
                        tracked_box
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    roi_gray = gray[
                        y1:y2,
                        x1:x2
                    ]

                    corners = (
                        cv2.goodFeaturesToTrack(
                            roi_gray,
                            maxCorners=20,
                            qualityLevel=0.3,
                            minDistance=7
                        )
                    )

                    if corners is not None:

                        corners[:, 0, 0] += x1
                        corners[:, 0, 1] += y1

                        track_points = corners

                else:

                    lost_frames += 1

                    if (
                        lost_frames >
                        MAX_LOST_FRAMES
                    ):

                        locked = False
                        tracked_box = None
                        tracked_class = None
                        track_points = None

            # ------------------------------------------------
            # INITIAL LOCK
            # ------------------------------------------------

            else:

                for det, class_name in detections:

                    if (
                        iou(
                            det,
                            manual_box
                        ) > 0.1
                    ):

                        tracked_box = det
                        tracked_class = class_name

                        locked = True

                        lost_frames = 0

                        x1, y1, x2, y2 = map(
                            int,
                            tracked_box
                        )

                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        roi_gray = gray[
                            y1:y2,
                            x1:x2
                        ]

                        corners = (
                            cv2.goodFeaturesToTrack(
                                roi_gray,
                                maxCorners=20,
                                qualityLevel=0.3,
                                minDistance=7
                            )
                        )

                        if corners is not None:

                            corners[:, 0, 0] += x1
                            corners[:, 0, 1] += y1

                            track_points = corners

                        break

    # =====================================================
    # TARGET GPS UPDATE
    # =====================================================

    if (
        locked and
        tracked_box is not None and
        tracked_class is not None
    ):

        target_gps = estimate_target_position(
            tracked_box,
            tracked_class
        )

    # =====================================================
    # DRAW TRACKING
    # =====================================================

    if locked and tracked_box is not None:

        x1, y1, x2, y2 = map(
            int,
            tracked_box
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"TRACK {tracked_class}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        if target_gps is not None:

            cv2.putText(
                frame,
                f"DIST "
                f"{target_gps['distance']:.1f}m",
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"BRG "
                f"{target_gps['bearing']:.1f}",
                (x1, y1 - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{target_gps['lat']:.6f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{target_gps['lon']:.6f}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # Optical flow points
        if track_points is not None:

            for point in track_points:

                px, py = point.ravel()

                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    3,
                    (255, 0, 0),
                    -1
                )

    else:

        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_w,
             box_y + box_h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FREE",
            (box_x, box_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # =====================================================
    # TELEMETRY DISPLAY
    # =====================================================

    if drone_lat is not None:

        cv2.putText(
            frame,
            f"ALT "
            f"{drone_alt:.1f}m",
            (20, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"HDG "
            f"{drone_heading:.1f}",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Manual movement
    if not locked:

        if key == ord('w'):

            box_y = max(
                0,
                box_y - MOVE_SPEED
            )

        elif key == ord('s'):

            box_y = min(
                height - box_h,
                box_y + MOVE_SPEED
            )

        elif key == ord('a'):

            box_x = max(
                0,
                box_x - MOVE_SPEED
            )

        elif key == ord('d'):

            box_x = min(
                width - box_w,
                box_x + MOVE_SPEED
            )

    # Reset tracking
    if key == ord('t'):

        locked = False

        tracked_box = None

        tracked_class = None

        track_points = None

        box_x, box_y = center_frame()

    prev_gray = gray.copy()

    frame_count += 1

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()
