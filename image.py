import cv2
import sys
from correction import correct_eyes_face


#Images paths as arguments
argv = sys.argv

args = len(argv)-1
for arg in range(args):
    path = argv[arg+1]
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    correct_eyes_face(image, True)

    cv2.imshow(path, image)

#Press any key to close windows
cv2.waitKey(0)