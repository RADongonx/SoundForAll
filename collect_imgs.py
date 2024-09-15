import os
import cv2

DATA_DIR = './hand_detective_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels = ["Yes", "No", "Thank You", "Sorry", "Hello", 
          "I Love You", "Goodbye", "Please", "Love", "Family"]

number_of_classes = len(labels)  
dataset_size = 200

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    label_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f'Collecting data for class {j}: {labels[j]}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f'Class: {labels[j]}. Press "Q" to start!', 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f'Capturing {labels[j]}: Image {counter}', 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(label_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
