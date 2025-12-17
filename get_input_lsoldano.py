"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program serves as a utility to obtain video input from 
    either a webcam or a video file located in the current directory. 
    It prompts the user to select the desired input method and returns 
    a video capture object for further processing.

Assignment Information:
    Assignment:     Individual Project 
    Team ID:        LC5 - 13
    Author:         Lucas, lsoldano@purdue.edu
    Date:           12/9/2025

Contributors:
    ChatGPT: Assisted with code structure and logic implementation.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
from pathlib import Path
import cv2

def get_vid_input():

    while True:
        directory_path = Path('.')

        cap_method = input('Select video source -> [1] Webcam | [2] Video File: ')

        if cap_method == '1':
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
            # Webcam check
            # Checks for error with webcam
            if not cap.isOpened():
                print("Could not open video device!")
                return None
            return cap

        if cap_method == '2':
            while True:
                vid_list = list(directory_path.glob('*.mov')) + list(directory_path.glob('*.mp4')) + list(directory_path.glob('*.avi'))
                print('\nAvailable video files:')
                j = 1
                for v in vid_list:
                    print(f'{j}- {v.name}')
                    j = j+1
                vid_selection = int(input('\nSelect a video file by number: '))
                if 1 <= vid_selection <= len(vid_list):
                    selected_video = vid_list[vid_selection - 1]
                    cap = cv2.VideoCapture(selected_video)
                    return cap
                else:
                    print("Invalid selection, please try again.\n")
                    
            
        else:
            print("Invalid input, please try again.\n")
