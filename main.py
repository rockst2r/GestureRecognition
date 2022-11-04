import cv2
import mediapipe as mp

# access camera
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Capture', frame)
        
    if cv2.waitKey(20) == ord('q'):
        break