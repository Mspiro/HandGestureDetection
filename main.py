import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic  # holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities


def mediapipe_detection(image, model):
    # Color Conversion BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # Image is now writable
    # Color Conversion RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS)  # draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw left_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw right_hand connections


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


# cap = cv2.VideoCapture(0)
# # set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()  # reading the video frames from camera
#         # make detaction
#         image, results = mediapipe_detection(frame, holistic)

#         # draw landmarks
#         draw_styled_landmarks(image, results)
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         # show to screen
#         cv2.imshow('Opencv Feed', image)

#         # braking the video loop using 'q'
#         if cv2.waitKey(10) & 0xff == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()


# print(results.face_landmarks.landmark)
# print(results.left_hand_landmarks.landmark)
# print(results.right_hand_landmarks.landmark)
# print(results.pose_landmarks.landmark)
#
# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)

# the following line produce the same result like the above for loop and this line is use to getting he all possitions
# of pose and hand array

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, right_hand])


# result_test = extract_keypoints(results)
# np.save('0', result_test)

# print(len(extract_keypoints(results)))

# print(len(pose))
# print(len(face))
# print(len(left_hand))
# print(len(right_hand))


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Action that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# 30 videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


#  Training model

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

# cap = cv2.VideoCapture(0)
# # set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # while cap.isOpened():
#     # loop through actions
#     for action in actions:
#         # loop through sequence or video
#         for sequence in range(no_sequences):
#             # loop through video length or sequence length
#             for frame_num in range(sequence_length):
#                 # reading the video frames/feed from camera
#                 ret, frame = cap.read()
#                 # make detaction
#                 image, results = mediapipe_detection(frame, holistic)
#                 print(results)
#                 # draw landmarks
#                 draw_styled_landmarks(image, results)
#                 plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#                 # Applay wait logic
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, 'Collectiong frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.waitKey(200)
#                 else:
#                     cv2.putText(image, 'Collectiong frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#
#                 # New Export Kelypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
#
#                 # show to screen
#                 cv2.imshow('Opencv Feed', image)
#
#                 # braking the video loop using 'q'
#                 if cv2.waitKey(10) & 0xff == ord('q'):
#                     break
#     cap.release()
#     cv2.destroyAllWindows()
#


label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(
                sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)
# print(np.array(labels).shape)

x = np.array(sequences)
# print(x.shape)
y = to_categorical(labels).astype(int)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

# print(x_test.shape)
# print(x_train.shape)
# print(y_test.shape)
# print(y_train.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# res = [.7, 0.2, 0.1]
# print(actions[np.argmax(res)])

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# training model with sequential nuaral network

# model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

# model.summary()

res = model.predict(x_test)
# print(np.sum(res[0]))
print(actions[np.argmax(res[0])])

actions[np.argmax(y_test[0])]   


model.save('action.h5')
# del model

model.load_weights('action.h5')

yhat = model.predict(x_train)

ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(ytrue)
print(yhat)

mcm = multilabel_confusion_matrix(ytrue, yhat)
print(mcm)
a = accuracy_score(ytrue, yhat)
print(a)


sequence = []
sentence = []
threshold = 0.4


cap = cv2.VideoCapture(0)
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # reading the video frames from camera
        # make detaction
        image, results = mediapipe_detection(frame, holistic)

        # draw landmarks
        draw_styled_landmarks(image, results)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # result_test = extract_keypoints(results)
        # np.save('0', result_test)
        # prediction logic
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res)
            print(actions[np.argmax(res)])

        # visualization logic
        # try:
        #     if res[np.argmax(res)] > threshold:
        #         if len(sentence) > 0:
        #             print(sentence)
        #             if actions[np.argmax(res)] != sentence[-1]:
        #                 sentence.insert(-1, actions[np.argmax(res)])
        #             else:
        #                 sentence.insert(-1, actions[np.argmax(res)])
        #             print(sentence)
        #     if len(sentence) > 5:
        #         sentence = sentence[-5:]
        #     cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        #     cv2.putText(image, ' ' .join(sentence), (3, 30),
        #             cv2.FONT_HERSHE_SUMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # except:
        #     pass

        # print(np.argmax(res) + " T: " + threshold)

        # if res[np.argmax(res)] > threshold:
        #     print("Here")
        #         if len(sentence) > 0:
        #             print(sentence)
        #             if actions[np.argmax(res)] != sentence[-1]:
        #                 sentence.insert(-1, actions[np.argmax(res)])
        #             else:
        #                 sentence.insert(-1, actions[np.argmax(res)])
        #             print(sentence)

        # if len(sentence) > 5:
        #     sentence = sentence[-5:]
        # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(image, ' ' .join(sentence), (3, 30),
        #     cv2.FONT_HERSHE_SUMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # # show to screen
        cv2.imshow('Opencv Feed', image)

        # braking the video loop using 'q'
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

