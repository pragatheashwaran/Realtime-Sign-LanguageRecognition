import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# --- CONFIG ---
MODEL_PATH = './model.p'
CAMERA_INDEX = 0

if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Please train the classifier first.")
    exit()

# --- LOAD MODEL ---
model_dict = pickle.load(open(MODEL_PATH, 'rb'))
model = model_dict['model']

# --- SETUP CAMERA ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

print("Starting Inference... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame to match data collection
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        all_landmarks = []
        box_coords = []
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

        for hand_landmarks in sorted_hands[:2]:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            box_coords.append((int(min(x_) * W) - 10, int(min(y_) * H) - 10, 
                               int(max(x_) * W) + 10, int(max(y_) * H) + 10))
            
            min_x_h, min_y_h = min(x_), min(y_)
            max_x_h, max_y_h = max(x_), max(y_)
            scale_ref = max(max_x_h - min_x_h, max_y_h - min_y_h, 1e-6)
            
            for i in range(len(hand_landmarks.landmark)):
                all_landmarks.append((hand_landmarks.landmark[i].x - min_x_h) / scale_ref)
                all_landmarks.append((hand_landmarks.landmark[i].y - min_y_h) / scale_ref)

        while len(all_landmarks) < 84:
            all_landmarks.append(0.0)

        try:
            # Predict
            pred_proba = model.predict_proba([np.asarray(all_landmarks)])
            confidence = np.max(pred_proba)
            
            if confidence > 0.4: # Moderate threshold
                prediction = model.predict([np.asarray(all_landmarks)])
                predicted_word = str(prediction[0])
            else:
                predicted_word = ""
    
            # Draw labels for all detected boxes
            for (x1, y1, x2, y2) in box_coords:
                if predicted_word != "":
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, predicted_word, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        except Exception as e:
            print(f"Prediction Error: {e}")

    cv2.imshow('Sign Language Word Reconignition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
