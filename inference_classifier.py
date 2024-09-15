import cv2
import mediapipe as mp
import numpy as np
import pickle
import pygame
import threading

MODEL_FILE = 'hand_detective_model.p'
LABELS = ["Yes", "No", "Thank You", "Sorry", "Hello", 
          "I Love You", "Goodbye", "Please", "Love", "Family"]
SOUND_FILES = {
    "Yes": "sounds/yes.mp3",
    "No": "sounds/no.mp3",
    "Thank You": "sounds/thank_you.mp3",
    "Sorry": "sounds/sorry.mp3",
    "Hello": "sounds/hello.mp3",
    "I Love You": "sounds/i_love_you.mp3",
    "Goodbye": "sounds/goodbye.mp3",
    "Please": "sounds/please.mp3",
    "Love": "sounds/love.mp3",
    "Family": "sounds/family.mp3"
}

pygame.mixer.init()

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_label = None
sound_thread = None

def play_sound(label):
    if label in SOUND_FILES:
        pygame.mixer.music.load(SOUND_FILES[label])
        pygame.mixer.music.play()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
     
    frame = cv2.flip(frame, 1)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    if len(landmarks) == 63:  
        landmarks.extend([0.0] * 63)  
    if len(landmarks) == 126: 
        prediction = model.predict([landmarks])[0]
        predicted_label = LABELS[prediction]
        cv2.putText(frame, predicted_label, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        if predicted_label != last_label:
            last_label = predicted_label
            if sound_thread is None or not sound_thread.is_alive():
                sound_thread = threading.Thread(target=play_sound, args=(predicted_label,))
                sound_thread.start()
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.quit()
