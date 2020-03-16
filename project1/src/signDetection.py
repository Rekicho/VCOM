import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from utils import *

# Detects all circles of a given color
def detectCircles(img, color):
    img = removeAllButOneColor(img,color)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low=np.array([1])
    high=np.array([255])
    mask = cv2.inRange(gray, low, high)
    gray[mask > 0] = 255
    kernel = np.ones((15,15),np.uint8)
    gray = cv2.dilate(gray,kernel,iterations = 1)
    gray = cv2.erode(gray,kernel,iterations = 1)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=50, maxRadius=250)
    if not(circles is None):
        circles = np.uint16(np.around(circles))
    return circles, gray

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

def detectStop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==8:
            cv2.drawContours(img,[cnt],0,(255,0,0),-1)

    return img

def detectStopText(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low=np.array([150])
    high=np.array([255])
    mask = cv2.inRange(gray, low, high)
    gray[mask > 0] = 0
    # print(pytesseract.image_to_string(gray, lang='eng', config='--psm 10 --oem 3'))
    d = pytesseract.image_to_data(gray, lang='eng', config='--psm 10 --oem 3', output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if d['text'][i] == 'STOP':
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img