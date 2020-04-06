import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from utils import *

class Detector:
    def __init__(self, img):
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.detected = {}
        self.processedImg = np.zeros([h,w,3], dtype=np.uint8)
        self.redProcessed = np.zeros([h,w,3], dtype=np.uint8)
        self.blueProcessed = np.zeros([h,w,3], dtype=np.uint8)

    def getDetected(self):
        return self.detected

    def detectAll(self):
        self.detectCircles("red")
        self.detectCircles("blue")

    def print2(self, img):
        cv2.imshow("Signs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process(self):
        # self.print2(self.blueProcessed)
        # self.print2(self.redProcessed)
        temp = cv2.bitwise_or(self.blueProcessed, self.redProcessed)
        self.processedImg = temp.copy()
        self.print2(self.processedImg)

    def detectCircles(self, color):
        img = self.processedImg
        img = removeAllButOneColor(img,color)
        print("isolating " + color)
        # self.print2(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        low=np.array([1])
        high=np.array([255])
        mask = cv2.inRange(gray, low, high)
        gray[mask > 0] = 255
        kernel = np.ones((15,15),np.uint8)
        gray = cv2.dilate(gray,kernel,iterations = 1)
        gray = cv2.erode(gray,kernel,iterations = 1)
        # self.print2(gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=100, minRadius=0, maxRadius=0)
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

    def detectTriangles(self, color):
        img = self.img
        img = removeAllButOneColor(img,color)
        cv2.imshow("White", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        # ret, thresh = cv2.threshold(gray, 127, 255, 1)
        #cv2.imshow("thresh", thresh)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        triangles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            print("Shapes: " + str(len(approx)))
            if len(approx) == 3:
                triangle = [(approx[0][0][0],approx[0][0][1]),
                            (approx[1][0][0],approx[1][0][1]),
                            (approx[2][0][0],approx[2][0][1])]
                center = (int(round((triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3)),
                          int(round((triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3)))
                if calcArea(triangle) > 0:
                    centers.append(center)
                    triangles.append(triangle)
                    # triangles.append([cnt])
        trianglesObj = {
            "info": triangles,
            "debugImg": gray,
            "coordText": centers,
            "text": "T"
        }
        # print(trianglesObj)
        self.detected["t"] = trianglesObj
        return contours, gray, triangles

    def detectRectangles(self):
        img = self.img
        # img = removeAllButOneColor(img,"blue")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 1)
        contours, h = cv2.findContours(thresh, 1, 2)
        rectangles = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                rectangles.append([cnt])
        RectanglesObj = {
            "info": contours,
            "debugImg": thresh,
            "coordText": rectangles,
            "text": "R"
        }
        
        self.detected["r"] = RectanglesObj
        return contours, thresh, rectangles