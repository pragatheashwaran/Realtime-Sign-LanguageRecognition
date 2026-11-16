import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

cap = cv2.VideoCapture(0)

print("Diagnostic: Testing Landmarks and Prediction...")
print("Press 'q' once you see your hand tracked to see the raw data.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        
        print(f"\nDetected {len(sorted_hands)} hands")
        
        for hand_idx, hand_landmarks in enumerate(sorted_hands[:2]):
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            
            # Print first 2 relative landmarks of the first hand for verification
            if hand_idx == 0:
                print(f"Hand 0 - First 2 relative landmarks: ({x_[0]-min_x:.4f}, {y_[0]-min_y:.4f}), ({x_[1]-min_x:.4f}, {y_[1]-min_y:.4f})")

            for i in range(len(hand_landmarks.landmark)):
                all_landmarks.append(hand_landmarks.landmark[i].x - min_x)
                all_landmarks.append(hand_landmarks.landmark[i].y - min_y)

        while len(all_landmarks) < 84:
            all_landmarks.append(0.0)

        pred_proba = model.predict_proba([np.asarray(all_landmarks)])
        confidence = np.max(pred_proba)
        
        if confidence > 0.4:
            prediction = model.predict([np.asarray(all_landmarks)])
            print(f"Prediction: {prediction[0]} (Confidence: {confidence:.2f})")
        else:
            print(f"Unknown sign (Confidence: {confidence:.2f} too low)")
        
    cv2.imshow('Debug', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
