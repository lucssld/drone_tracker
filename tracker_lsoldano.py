'''
This program detects and tracks selected objects across the video frame using IOU selection. 
This tracks the object by finding the detection box with most overlap to the previous frame's 
tracked detection. This method does not work well for fast-moving frames, but it is extremely 
lightweight and works well depending on the selected model. Through testing, I found the best
model to be visDrone.

Users can move a targeting box over the object they would like to track using >>>'W', 'A', 'S', 'D'<<<
and can press >>>'T'<<< to reset the window. Once an object is tracked, the window turns red and
should follow the target. To quit, press >>>'Q'<<<

This program implements CUDA as a way to process detections using the computer's GPU. This results
in faster, smoother results.

Make sure to include model files (*.pt) within the same file as the program.
'''

import cv2
import numpy
from ultralytics import YOLO
from pathlib import Path

print('\n-----SETUP-----\n')

directory_path = Path('.') # Sets path to program's folder

#-----Model selection-----
while True:
    model_list = list(directory_path.glob('*.pt'))
    print('\nAvailable YOLO models:')
    j = 1
    for m in model_list: # prints out numbered list of files
        print(f'{j}- {m.name}')
        j = j+1
    model_selection = int(input('\nSelect a model file by number: '))
    # Ensures valid selection
    if 1 <= model_selection <= len(model_list):
        selected_model = model_list[model_selection - 1]
        break
    else:
        print("Invalid selection, please try again.\n")

#selected_model = 'visDrone.pt'  # >>>Default model<<<

model = YOLO(selected_model)

#-----Video source selection-----


while True:

    cap_method = input('Select video source -> [1] Webcam | [2] Video File: ')

    if cap_method == '1':
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
        # Webcam check
        # Checks for error with webcam
        if not cap.isOpened():
            print("Could not open video device!")
            cap = None
        break

    if cap_method == '2':
        # Error loop, runs until valid video file is selected
        while True:
            vid_list = list(directory_path.glob('*.mov')) + list(directory_path.glob('*.mp4')) + list(directory_path.glob('*.avi'))
            print('\nAvailable video files:')
            j = 1
            for v in vid_list: # prints out numbered list of files
                print(f'{j}- {v.name}')
                j = j+1
            vid_selection = int(input('\nSelect a video file by number: '))
            # Ensures valid selection
            if 1 <= vid_selection <= len(vid_list):
                selected_video = vid_list[vid_selection - 1]
                cap = cv2.VideoCapture(selected_video)
                break
            else:
                print("Invalid selection, please try again.\n")            
    else:
        print("Invalid input, please try again.\n")

if cap is None:
    print("Could not obtain video source. Exiting.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer setup for video output
out = cv2.VideoWriter('output_manual_lock.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#-----Target box size selection-----
user = input("\nEnter targeter size | [1] Small (50x50) | [2] Medium (100x100) | [3] Large (150x150): ")
if user == '1':
    box_w, box_h = 50, 50
elif user == '2':
    box_w, box_h = 100, 100
elif user == '3':
    box_w, box_h = 150, 150
else:
    print("Invalid input, defaulting to Medium (100x100).")
    box_w, box_h = 100, 100

#Enable GPU if available
try:
    model.to('cuda')
    print("Using GPU for inference.")
except:
    print("***GPU not available, using CPU***")

input("Press Enter to start video processing...")

# Targeter Parameters
MOVE_SPEED = 15  # pixels per key press
MAX_LOST_FRAMES = 30 # Lost frames before unlocking
IOU_THRESHOLD = 0.3 # Overlap threshold to lock

#-----Box setup-----

# Creates box centered at center of frame with dimensions selected by user
box_x = width // 2 - box_w // 2
box_y = height // 2 - box_h // 2
tracked_box = None
lost_frames = 0
locked = False

#Computes overlap between tracker and detections
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

#Return bounding box [x1, y1, x2, y2]
def make_box(x, y, w, h):
    return [x, y, x + w, y + h]

#Brings targeter to center of frame
def center_frame():
    return width // 2 - box_w // 2, height // 2 - box_h // 2

#-----Processing loop-----
while True:
    ret, frame = cap.read()

    # End of video check
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    detections = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) > 0 else []

    #make manual box
    manual_box = make_box(box_x, box_y, box_w, box_h)

    # Tracking logic
    if locked and tracked_box is not None:
        matched = False
        for det in detections:
            if iou(det, tracked_box) > IOU_THRESHOLD:
                tracked_box = det
                matched = True
                lost_frames = 0
                break
        #adds to lost frames if no match found
        if not matched:
            lost_frames += 1

        # Lost target for too long — unlock
        if lost_frames > MAX_LOST_FRAMES:
            locked = False
            tracked_box = None
    else:
        # When unlocked, check if any detection overlaps manual box
        for det in detections:
            if iou(det, manual_box) > 0.1:
                tracked_box = det
                locked = True
                lost_frames = 0
                break
    # Draw boxes on screen
    if locked and tracked_box is not None:

        x1, y1, x2, y2 = map(int, tracked_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "TRACK", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Draw the movable manual box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        cv2.putText(frame, "FREE", (box_x, box_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tracker", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # --- Controls ---
    if key == ord('q'):
        break

    if not locked:
        if key == ord('w'):
            box_y = max(0, box_y - MOVE_SPEED)
        elif key == ord('s'):
            box_y = min(height - box_h, box_y + MOVE_SPEED)
        elif key == ord('a'):
            box_x = max(0, box_x - MOVE_SPEED)
        elif key == ord('d'):
            box_x = min(width - box_w, box_x + MOVE_SPEED)

    if key == ord('t'):
        # Unlock and recenter
        locked = False
        tracked_box = None
        box_x, box_y = center_frame()

cap.release()
out.release()
cv2.destroyAllWindows()
