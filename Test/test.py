# from unittest import result
from re import I
import cv2
import numpy as np
import os
from matplotlib import image, pyplot as plt
import time
import mediapipe as mp


mp_holistic = mp.solutions.holistic  # Holistic Model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities


def mediapipe_detection(image, model):
    # COLOR Conversion BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Image is not writeable
    resutls = model.process(image)  # Make Prediction
    image.flags.writeable = True  # Image is writeable
    # Color Conversion RGR 2 BGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, resutls


def draw_landmarks(image, results):
    # Draw Face connections
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    # Draw Pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw Left Hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw Right Hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    # draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # draw left_hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # draw right_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read feed
        ret, frame = cap.read()

        # Make Prediction
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        #  'results' variable have all data realted to face, hand, pose

        # Draw Landmarks
        draw_styled_landmarks(image, results)

        # Show the window
        cv2.imshow('OpenCV Feed', image)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Break the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
