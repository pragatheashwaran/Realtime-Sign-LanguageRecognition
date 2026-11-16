import mediapipe as mp
print("Mediapipe version:", mp.__version__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
print("Mediapipe hands initialized")
