import cv2
import numpy as np
from flask import Flask, Response, request
from queue import Queue
from threading import Thread
import mediapipe as mp
import tensorflow as tf
import logging
import time
from keras.models import load_model
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Check for GPU support
print("Built with GPU support:", tf.test.is_built_with_gpu_support())
print("GPU devices available:", tf.config.list_physical_devices('GPU'))

# Initialize MediaPipe hands, pose, and face mesh detectors
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Prepare drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize the image queue and thread
image_queue = Queue(maxsize=1)
def translate_to_origin(point_a, point_b, point_c):
    return point_b - point_a, point_c - point_a

def rotate_matrix_z(angle):
    """
    Returns a rotation matrix around the z-axis by the given angle in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def rotate_matrix_x(angle):
    """
    Returns a rotation matrix around the x-axis by the given angle in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotate_matrix_y(angle):
    """
    Returns a rotation matrix around the y-axis by the given angle in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def compute_transform_matrix(point_a, point_b, point_c):
    # Rotate around the Z-axis
    angle_z = np.arctan2(point_b[0], point_b[1])
    rotation_matrix_z = rotate_matrix_z(angle_z)
    rotated_b = rotation_matrix_z @ point_b
    
    #Rotate around the X-axis
    angle_x = -np.arctan2(rotated_b[2], rotated_b[1])
    rotation_matrix_x = rotate_matrix_x(angle_x)
    rotated_b = rotation_matrix_x @ rotated_b.transpose()
    
    # Rotate translated_c around Z and X axes
    rotated_c = rotation_matrix_x @ (rotation_matrix_z @ point_c)
    
    # Rotate around the Y-axis
    angle_y = np.arctan2(rotated_c[2], rotated_c[0])
    rotation_matrix_y = rotate_matrix_y(angle_y)
    rotated_c = rotation_matrix_y@rotated_c
    
    # Combine transformations
    rotation_matrix = rotation_matrix_y @ rotation_matrix_x @ rotation_matrix_z 
    
    return rotation_matrix

def process_frame(frame, model, train_labels):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand, body, and face poses in the frame
    hand_results = hands.process(rgb_frame)
    #face_results = face_mesh.process(rgb_frame)

    # Draw the hand pose landmarks and connections
    landmarks = np.empty((21,3))
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if hand_results.multi_hand_landmarks:
        #print(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.empty((21,3))
            i = 0
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                coord = [x, y, z]
                landmarks[i] = coord
                i += 1

            # Flip landmarks for the right hand
            left_bool = hand_results.multi_handedness[0].classification[0].label == "Left"
            if not left_bool:
                landmarks[:, 0] = -landmarks[:, 0]

            # Translate landmarks to origin
            landmarks = landmarks - landmarks[0]

            point_a = landmarks[0]
            point_b = landmarks[5]
            point_c = landmarks[17]

            # Compute the transformation matrix
            tf_mat = compute_transform_matrix(point_a, point_b, point_c)
            i = 0
            for p in landmarks:
                T = tf_mat @ p
                landmarks[i] = T
                i += 1

            # Normalize the landmark coordinates
            bool_arr = abs(landmarks) < (10**-15)
            landmarks[bool_arr] = 0
            x = landmarks[:,0]
            x = np.interp(x, (x.min(), x.max()), (0, +1))
            y = landmarks[:,1]
            y = np.interp(y, (y.min(), y.max()), (0, +1))
            z = landmarks[:,2]
            z = np.interp(z, (z.min(), z.max()), (0, +1))
            landmarks = np.asarray([x,y,z]).transpose()

        predictions = model.predict(np.asarray([landmarks]), batch_size = 1)
        print(alphabet[np.argmax(predictions)])

    return frame

def process_images():
    model = load_model('ASL_Model')
    train_labels = np.load("data/train_labels.npy")
    #conv_str = 
    while True:
        prev_time = time.time()
        img = image_queue.get()
        processed_frame = process_frame(img, model, train_labels)

        # Calculate and display the frame rate
        current_time = time.time()
        fps = 1 / (current_time - prev_time)

        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (processed_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Webcam', processed_frame)
        cv2.waitKey(1)

image_thread = Thread(target=process_images)
image_thread.daemon = True
image_thread.start()

@app.route('/video_feed', methods=['POST'])
def video_feed():
    img_data = request.data
    img = cv2.imdecode(np.frombuffer(img_data, dtype=np.byte), cv2.IMREAD_COLOR)

    # Add the image to the queue for processing and display
    image_queue.put(img)

    return Response(status=200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)