import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100

# Mediapipe for monitoring
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Try index 0, 1, or 2.")
    exit()

print("--- Sign Language Word Collector ---")
print("Type the word you want to collect data for and press Enter.")
print("Type 'EXIT' to stop collecting.")

while True:
    word = input("\n[STEP 1] Type the word name (e.g., HELLO) and press ENTER: ").strip().upper()
    
    if word == 'EXIT':
        break
    
    if not word:
        print("!! WARNING: You didn't type anything. Please enter a word name.")
        continue
    
    word_path = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_path):
        os.makedirs(word_path)

    print(f'-- OK: Ready for "{word}" --')
    print('Look at the camera window. Press "Q" when you are IN POSITION to start.')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        
        # --- NEW: DRAW HANDS FOR MONITORING ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # -------------------------------------

        cv2.putText(display_frame, f'Collecting: {word}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, 'Press "Q" to start', (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add background to text for better visibility
        cv2.rectangle(display_frame, (15, 65), (300, 110), (0,0,0), -1)
        cv2.putText(display_frame, 'Press Q to Capture', (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Collector', display_frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start capturing
    print(f'-> Capturing {dataset_size} images now! Keep signing...')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror while saving so inference matches collection
        frame = cv2.flip(frame, 1)
            
        cv2.imshow('Collector', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(word_path, f'{counter}.jpg'), frame)
        
        # Periodic status
        if (counter + 1) % 25 == 0:
            print(f'... {counter + 1}%')
            
        counter += 1

    print(f'SUCCESS: Data for "{word}" saved in folder: {word_path}')
    print("---------------------------------------------------")

cap.release()
cv2.destroyAllWindows()
print("\nCollection finished. Next step: Run 'python create_dataset.py'")
