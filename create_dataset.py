import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7, max_num_hands=2)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            all_landmarks = []
            
            # Sort hands by X to keep Left Hand vs Right Hand consistent in the data vector
            sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
            
            for hand_landmarks in sorted_hands[:2]:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                max_x, max_y = max(x_), max(y_)
                
                # Use the larger dimension for scale normalization to preserve aspect ratio
                scale_ref = max(max_x - min_x, max_y - min_y, 1e-6)
                
                for i in range(len(hand_landmarks.landmark)):
                    all_landmarks.append((hand_landmarks.landmark[i].x - min_x) / scale_ref)
                    all_landmarks.append((hand_landmarks.landmark[i].y - min_y) / scale_ref)
            
            # Pad if only one hand is detected
            while len(all_landmarks) < 84:
                all_landmarks.append(0.0)

            data.append(all_landmarks)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
