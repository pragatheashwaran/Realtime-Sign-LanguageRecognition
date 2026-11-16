import pickle
import numpy as np
import os
import mediapipe as mp

MODEL_PATH = 'model.p'

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"FAILED: {MODEL_PATH} not found.")
        return
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict['model']
        print("SUCCESS: Model loaded.")
        
        # Test a dummy prediction with 84 features
        dummy_input = np.zeros((1, 84))
        prediction = model.predict(dummy_input)
        print(f"SUCCESS: Dummy prediction: {prediction}")
        
        if hasattr(model, 'classes_'):
            print(f"Classes: {model.classes_}")
    except Exception as e:
        print(f"FAILED: Error: {e}")

def test_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        print("SUCCESS: Mediapipe initialized.")
    except Exception as e:
        print(f"FAILED: Mediapipe error: {e}")

if __name__ == "__main__":
    print("--- Diagnostic Script ---")
    test_model()
    test_mediapipe()
