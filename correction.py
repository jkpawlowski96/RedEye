import cv2
import numpy as np
from detection import detect_eyes_face, detect_eyes, detect_faces


def correct_eyes(image, eyes):
    img = image
    if eyes is None:
        return None
    for x, y, w, h in eyes:
        # Crop the eye region
        img_eye = image[y:y + h, x:x + w]

        # split the images into 3 channels
        b, g, r = cv2.split(img_eye)

        # Add blue and green channels
        bg = cv2.add(b, g)

        # threshold the mask based on red color and combination ogf blue and gree color
        mask = ((r > (bg - 20)) & (r > 50)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)  # It return contours and Hierarchy

        # find contour with max Area
        max_area = 0
        max_cont = None
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > max_area:
                max_area = area
                max_cont = cont
        mask = mask * 0  # Reset the mask image to complete black image
        # draw the biggest contour on mask
        cv2.drawContours(mask, [max_cont], 0, 255, -1)
        # Close the holes to make a smooth region
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5)))
        mask = cv2.dilate(mask, (3, 3), iterations=3)

        # --The information of only red color is lost ,
        # So we fill the mean of blue and green color in all three channels(BGR) to maintain the texture
        mean = bg / 2

        # Fill this black mean value to masked image
        mean = cv2.bitwise_and(mean.astype(np.uint8), mask)  # mask the mean image
        mean = cv2.cvtColor(mean, cv2.COLOR_GRAY2BGR)  # convert mean to 3 channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # convert mask to 3 channel
        eye = cv2.bitwise_and(~mask, img_eye) + mean  # Copy the mean color to masked region to color image
        img[y:y + h, x:x + w] = eye


#Find eyes in faces area and correct them
def correct_eyes_face(image, underline):
    img0 = image.copy()
    img = image
    #Faces
    faces = detect_faces(img, underline)
    for (x, y, w, h) in faces:
        face_area = img[y:y + h, x:x + w]
        face_area0 = img0[y:y + h, x:x + w]
        #Eyes in the face
        eyes = detect_eyes(face_area0, underline)
        correct_eyes(face_area, eyes)
    #Underline borders
    if underline:
        for (x, y, w, h) in faces:
            face_area = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
