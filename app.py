import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS

import sys

# Determine if we are running in a bundled executable
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    BASE_DIR = sys._MEIPASS
else:
    app = Flask(__name__)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CORS(app)

# --- CONFIG ---
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.4  # Moderate threshold for random forest predictions

# --- HELPERS ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"!! WARNING: {MODEL_PATH} not found. System will run without predictions.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    except ModuleNotFoundError as e:
        print("\n" + "!"*60)
        print(f"CRITICAL ERROR: {e}")
        print("This usually happens due to a NumPy version mismatch.")
        print("Try running the app using the virtual environment:")
        print("    .\\venv\\Scripts\\python.exe app.py")
        print("!"*60 + "\n")
        return None
    except Exception as e:
        print(f"!! Error loading model: {e}")
        return None

model = load_model()
if model:
    print("✓ Model successfully loaded from", MODEL_PATH)
    if hasattr(model, 'classes_'):
        print(f"  Trained signs: {list(model.classes_)}")
else:
    print("!! Application running in MONITORING MODE ONLY (No predictions).")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# --- GLOBAL STATE ---
current_prediction = "---"
current_confidence = 0.0
prediction_history = []
HISTORY_LENGTH = 10  # Number of frames to average/debounce

def gen_frames():
    global current_prediction, prediction_history, current_confidence
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Use the globally initialized 'hands' object for better performance
    # instead of creating a new one inside the loop
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_word = ""
        box_coords = []

        if results.multi_hand_landmarks:
            # Draw landmarks for visual feedback
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if model:
                all_landmarks = []
                # Sort hands by X to maintain consistency (Left-to-Right)
                sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

                for hand_landmarks in sorted_hands[:2]:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    
                    # Store box for drawing later
                    box_coords.append((int(min(x_) * W) - 10, int(min(y_) * H) - 10, 
                                     int(max(x_) * W) + 10, int(max(y_) * H) + 10))
                    
                    # Relative landmarks with scale normalization to preserve aspect ratio
                    min_x_h, min_y_h = min(x_), min(y_)
                    max_x_h, max_y_h = max(x_), max(y_)
                    scale_ref = max(max_x_h - min_x_h, max_y_h - min_y_h, 1e-6)
                    
                    for lm in hand_landmarks.landmark:
                        all_landmarks.append((lm.x - min_x_h) / scale_ref)
                        all_landmarks.append((lm.y - min_y_h) / scale_ref)

                # Ensure we always have 84 features (2 hands * 21 landmarks * 2 dimensions)
                while len(all_landmarks) < 84:
                    all_landmarks.append(0.0)

                try:
                    features = np.asarray(all_landmarks).reshape(1, -1)
                    
                    # Use predict_proba to get confidence scores
                    probabilities = model.predict_proba(features)[0]
                    max_confidence = np.max(probabilities)
                    predicted_class = model.classes_[np.argmax(probabilities)]
                    
                    # Only accept prediction if confidence is above threshold
                    if max_confidence >= CONFIDENCE_THRESHOLD:
                        raw_pred = str(predicted_class)
                    else:
                        raw_pred = ""  # Hide unknown/low-confidence signs
                    
                    # History buffer to stabilize predictions
                    prediction_history.append(raw_pred)
                    if len(prediction_history) > HISTORY_LENGTH:
                        prediction_history.pop(0)
                    
                    # Consensus logic: Only show sign if it's consistently predicted
                    from collections import Counter
                    counts = Counter(prediction_history)
                    most_common = counts.most_common(1)
                    
                    if most_common and most_common[0][0] != "" and most_common[0][1] >= (HISTORY_LENGTH * 0.4):
                        predicted_word = most_common[0][0]
                    else:
                        predicted_word = "" # Display nothing if no strong consensus
                    
                    current_confidence = max_confidence
                        
                    # Draw boxes and predicted text on frame (only if a valid sign is predicted)
                    if predicted_word != "":
                        color = (46, 204, 113)
                        for (x1, y1, x2, y2) in box_coords:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                            label = f"{predicted_word} ({int(max_confidence*100)}%)"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Prediction Error: {e}")
            else:
                predicted_word = "NO MODEL"
        else:
            # No hands detected, clear history faster
            if prediction_history:
                prediction_history.pop(0)
            predicted_word = ""

        current_prediction = predicted_word
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        "prediction": current_prediction,
        "confidence": round(current_confidence * 100, 1)
    })

@app.route('/status')
def status():
    classes = []
    if model and hasattr(model, 'classes_'):
        classes = model.classes_.tolist()
        
    return jsonify({
        "model_loaded": model is not None,
        "labels": classes
    })

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Only open browser if:
    # 1. We are NOT in the Flask reloader (WERKZEUG_RUN_MAIN is not set)
    # 2. OR we are running as a frozen executable (PyInstaller)
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    is_frozen = getattr(sys, 'frozen', False)

    if not is_reloader:
        Timer(1.5, open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
