import cv2
import sys
import numpy as np

FPS = 60

def openImage(name):
    return cv2.imread(name,cv2.IMREAD_COLOR)

def takeImageFromCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(True):
        ret, frame = cap.read()
        img = frame
        cv2.imshow('Camera',img)
        if cv2.waitKey(int(1000/FPS)) != -1:
            break
    cap.release()
    cv2.destroyAllWindows()
    return img

if len(sys.argv) > 1: 
    img = openImage(sys.argv[1])
else:
    img = takeImageFromCamera()
r = img.copy()
r[:, :, 1] = 0
r[:, :, 0] = 0
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow("Sign",img)
cv2.waitKey(0)
cv2.destroyAllWindows()