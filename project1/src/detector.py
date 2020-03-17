import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from utils import *

class Detector:
    def __init__(self, img):
        self.img = img
        self.detected = {}

    def getDetected(self):
        return self.detected

    def detectAll(self):
        self.detectCircles("red")
        self.detectCircles("blue")

    def detectCircles(self, color):
        img = self.img
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
        centers = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            centers = []
            for i in circles[0,:]:
                center = (i[0],i[1])
                centers.append(center)
            circlesObj = {
                "info": circles,
                "debugImg": gray,
                "coordText": centers,
                "text": color + "C"
            }
            self.detected["c-" + color] = circlesObj
        return circles, gray, centers

    def detectTriangles(self):
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 1)
        contours, h = cv2.findContours(thresh, 1, 2)
        triangles = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                triangles.append([cnt])
        trianglesObj = {
            "info": contours,
            "debugImg": thresh,
            "coordText": triangles,
            "text": "T"
        }
        self.detected["t"] = trianglesObj
        return contours, thresh, triangles