import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

DATA_DIR = './hand_detective_data'
PROCESSED_DATA_FILE = 'hand_detective_data.pickle'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

data = []
labels = []

for label_idx, label in enumerate(os.listdir(DATA_DIR)):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        if len(landmarks) == 63:  
            landmarks.extend([0.0] * 63) 
        
        if len(landmarks) == 126:  
            data.append(landmarks)
            labels.append(label_idx)

hands.close()

with open(PROCESSED_DATA_FILE, 'wb') as f:
    pickle.dump((np.array(data), np.array(labels)), f)
