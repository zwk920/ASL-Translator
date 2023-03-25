import cv2
import numpy as np
from flask import Flask, Response, request
from queue import Queue
from threading import Thread
import mediapipe as mp
import inspect
#import tensorflow as tf
# import OS module
import os

# Check for GPU support
#print("Built with GPU support:", tf.test.is_built_with_gpu_support())
#print("GPU devices available:", tf.config.list_physical_devices('GPU'))

# Initialize MediaPipe hands, pose, and face mesh detectors
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Prepare drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize the image queue and thread
image_queue = Queue(maxsize=1)
def process_frame(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand poses in the frame
    hand_results = hands.process(rgb_frame)

    # Draw the hand pose landmarks and connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                landmarks.append((x, y, z))
            print(len(landmarks))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame

def process_images():
    # folder path
    dir_path = 'D:/Desktop/hand data/train/'

    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith('.jpg'):
            pic =dir_path+file
            res.append(pic)
    
    #print(res)

    for name in res:
        img = cv2.imread(name)
        processed_frame = process_frame(img)

        cv2.imshow('Webcam', processed_frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    process_images()


