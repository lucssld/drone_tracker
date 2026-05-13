'''
Hybrid YOLO + Optical Flow Tracker
Optimized for low-power systems like Raspberry Pi

Features:
- Sparse YOLO detections
- Optical flow tracking between detections
- ROI-only detections for huge speed gains
- Manual target selection

Controls:
W A S D = Move targeting box
T = Reset tracking
Q = Quit
'''

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

print('\n-----SETUP-----\n')

directory_path = Path('.')

# =========================================================
# Model Selection
# =========================================================
while True:

    model_list = list(directory_path.glob('*.pt'))

    print('\nAvailable YOLO models:')

    for i, model_file in enumerate(model_list, start=1):
        print(f'{i} - {model_file.name}')

    model_selection = int(input('\nSelect model number: '))

    if 1 <= model_selection <= len(model_list):
        selected_model = model_list[model_selection - 1]
        break

    print("Invalid selection.\n")

model = YOLO(selected_model)

# =========================================================
# Video Source Selection
# =========================================================
selection = False

while not selection:

    cap_method = input(
        'Select video source -> [1] Webcam | [2] Video File: '
    )

    if cap_method == '1':

        cap = cv2.VideoCapture(0)

        # Lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not cap.isOpened():
            print("Could not open webcam.")
            exit()

        selection = True

    elif cap_method == '2':

        while True:

            vid_list = (
                list(directory_path.glob('*.mov')) +
                list(directory_path.glob('*.mp4')) +
                list(directory_path.glob('*.avi'))
            )

            print('\nAvailable video files:')

            for i, vid_file in enumerate(vid_list, start=1):
                print(f'{i} - {vid_file.name}')

            vid_selection = int(input('\nSelect video number: '))

            if 1 <= vid_selection <= len(vid_list):

                selected_video = vid_list[vid_selection - 1]

                cap = cv2.VideoCapture(str(selected_video))

                selection = True
                break

            print("Invalid selection.\n")

    else:
        print("Invalid input.\n")

# =========================================================
# Video Properties
# =========================================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    print("Invalid input. Defaulting to Medium.")
    box_w, box_h = 100, 100

input("Press Enter to start...")

# =========================================================
# Parameters
# =========================================================
MOVE_SPEED = 15

MAX_LOST_FRAMES = 30

IOU_THRESHOLD = 0.3

# YOLO runs every N frames
frame_skip = 10

# ROI sizes
ROI_PADDING_UNLOCKED = 150
ROI_PADDING_LOCKED = 100

# =========================================================
# Tracking State
# =========================================================
box_x = width // 2 - box_w // 2
box_y = height // 2 - box_h // 2

tracked_box = None

locked = False

lost_frames = 0

frame_count = 0

prev_gray = None

track_points = None

# =========================================================
# Utility Functions
# =========================================================
def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

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
        boxAArea + boxBArea - interArea
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
# Main Processing Loop
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray.copy()

    manual_box = make_box(
        box_x,
        box_y,
        box_w,
        box_h
    )

    # =====================================================
    # Optical Flow Tracking
    # =====================================================
    if (
        locked and
        tracked_box is not None and
        track_points is not None
    ):

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            track_points,
            None,
            winSize=(15, 15),
            maxLevel=2
        )

        if new_points is not None:

            good_new = new_points[status == 1]
            good_old = track_points[status == 1]

            if len(good_new) > 0:

                movement = good_new - good_old

                dx, dy = movement.mean(axis=0)

                tracked_box[0] += dx
                tracked_box[1] += dy
                tracked_box[2] += dx
                tracked_box[3] += dy

                track_points = good_new.reshape(-1, 1, 2)

            else:

                locked = False
                tracked_box = None
                track_points = None

    # =====================================================
    # YOLO Detection (ROI ONLY)
    # =====================================================
    if frame_count % frame_skip == 0:

        # -------------------------------------------------
        # Define ROI
        # -------------------------------------------------
        if locked and tracked_box is not None:

            tx1, ty1, tx2, ty2 = map(int, tracked_box)

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
                box_x - ROI_PADDING_UNLOCKED
            )

            roi_y1 = max(
                0,
                box_y - ROI_PADDING_UNLOCKED
            )

            roi_x2 = min(
                width,
                box_x + box_w + ROI_PADDING_UNLOCKED
            )

            roi_y2 = min(
                height,
                box_y + box_h + ROI_PADDING_UNLOCKED
            )

        # Crop ROI
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

            if boxes is not None and len(boxes) > 0:

                local_boxes = boxes.xyxy.cpu().numpy()

                # Convert ROI coords -> full frame coords
                for det in local_boxes:

                    det[0] += roi_x1
                    det[1] += roi_y1
                    det[2] += roi_x1
                    det[3] += roi_y1

                    detections.append(det)

            # ------------------------------------------------
            # Reacquire Existing Target
            # ------------------------------------------------
            if locked and tracked_box is not None:

                matched = False

                best_iou = 0

                best_det = None

                for det in detections:

                    overlap = iou(det, tracked_box)

                    if overlap > best_iou:

                        best_iou = overlap
                        best_det = det

                if (
                    best_det is not None and
                    best_iou > IOU_THRESHOLD
                ):

                    tracked_box = best_det

                    matched = True

                    lost_frames = 0

                    x1, y1, x2, y2 = map(
                        int,
                        tracked_box
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    roi_gray = gray[y1:y2, x1:x2]

                    corners = cv2.goodFeaturesToTrack(
                        roi_gray,
                        maxCorners=20,
                        qualityLevel=0.3,
                        minDistance=7
                    )

                    if corners is not None:

                        corners[:, 0, 0] += x1
                        corners[:, 0, 1] += y1

                        track_points = corners

                else:

                    lost_frames += 1

                    if lost_frames > MAX_LOST_FRAMES:

                        locked = False
                        tracked_box = None
                        track_points = None

            # ------------------------------------------------
            # Initial Target Lock
            # ------------------------------------------------
            else:

                for det in detections:

                    if iou(det, manual_box) > 0.1:

                        tracked_box = det

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

                        roi_gray = gray[y1:y2, x1:x2]

                        corners = cv2.goodFeaturesToTrack(
                            roi_gray,
                            maxCorners=20,
                            qualityLevel=0.3,
                            minDistance=7
                        )

                        if corners is not None:

                            corners[:, 0, 0] += x1
                            corners[:, 0, 1] += y1

                            track_points = corners

                        break

    # =====================================================
    # Draw Tracking
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
            "TRACK",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # Draw optical flow points
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
            (box_x + box_w, box_y + box_h),
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
    # Display
    # =====================================================
    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Manual movement
    if not locked:

        if key == ord('w'):
            box_y = max(0, box_y - MOVE_SPEED)

        elif key == ord('s'):
            box_y = min(
                height - box_h,
                box_y + MOVE_SPEED
            )

        elif key == ord('a'):
            box_x = max(0, box_x - MOVE_SPEED)

        elif key == ord('d'):
            box_x = min(
                width - box_w,
                box_x + MOVE_SPEED
            )

    # Reset tracking
    if key == ord('t'):

        locked = False

        tracked_box = None

        track_points = None

        box_x, box_y = center_frame()

    # Update previous frame
    prev_gray = gray.copy()

    frame_count += 1

# =========================================================
# Cleanup
# =========================================================
cap.release()

cv2.destroyAllWindows()