import cv2
import numpy as np
from detection import detect_eyes_face


cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #frameOut, eyes = red_eye.detect_eyes_face(frame, True)
    detect_eyes_face(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
