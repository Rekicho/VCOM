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

def detectCircles(img):
    #r = img.copy()
    #r[:, :, 1] = 0
    #r[:, :, 0] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return img

def detectTriangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==3:
            cv2.drawContours(img,[cnt],0,(0,255,0),-1)

    return img

def detectRectangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            cv2.drawContours(img,[cnt],0,(255,0,0),-1)

    return img

if len(sys.argv) > 1: 
    img = openImage(sys.argv[1])
    #img = detectTriangles(img)
    #img = detectRectangles(img)
    #img = detectCircles(img)
else:
    img = takeImageFromCamera()
cv2.imshow("Sign",img)
cv2.waitKey(0)
cv2.destroyAllWindows()