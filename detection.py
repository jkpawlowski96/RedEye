import cv2


def detect_faces(image, underline):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if not underline:
        img = image.copy()
    else:
        img = image
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return faces


def detect_eyes(image, underline):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    if not underline:
        img = image.copy()
    else:
        img = image
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes on face and show them
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return eyes


#Find eyes in faces
def detect_eyes_face(image):
    img = image
    #Faces
    faces = detect_faces(img, False)
    for (x, y, w, h) in faces:
        face_area = img[y:y + h, x:x + w]
        eyes = detect_eyes(face_area, False)

    # Underline borders
    for (x, y, w, h) in faces:
        face_area = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
