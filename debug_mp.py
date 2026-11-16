import mediapipe as mp
print("Imported mediapipe")
try:
    hands = mp.solutions.hands.Hands()
    print("Hands initialized")
except Exception as e:
    print(f"Error: {e}")
