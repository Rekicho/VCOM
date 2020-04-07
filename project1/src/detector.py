import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from utils import *

# Percentage of image size signs have to be to be considered
MINIMUM_SIGN_SIZE = 0.01 

class Detector:
    def __init__(self, img):
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.minimumSignSize = MINIMUM_SIGN_SIZE * h * w
        print("Minimum Sign Size: " + str(self.minimumSignSize))
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

    def detectHoughCircles(self, color):
        img = self.processedImg
        img = removeAllButOneColor(img,color)
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

    def detectCircles(self, color):
        img = self.img
        img = removeAllButOneColor(img,color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 1)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) > 8:
                circle = []
                for coords in approx:
                    circle.append((coords[0][0],coords[0][1]))
                if calculateArea(circle) >= self.minimumSignSize:
                    print("Circle: " + str(calculateArea(circle)))
                    center = getCenter(circle)
                    centers.append(center)
                    circles.append([cnt])
        circlesObj = {
            "info": circles,
            "debugImg": gray,
            "coordText": centers,
            "text": "c-" + color
        }
        self.detected["c-" + color] = circlesObj
        return contours, gray, circles


    def detectTriangles(self, color):
        img = self.img
        img = removeAllButOneColor(img,color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 1)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        triangles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                triangle = [(approx[0][0][0],approx[0][0][1]),
                            (approx[1][0][0],approx[1][0][1]),
                            (approx[2][0][0],approx[2][0][1])]
                if calculateArea(triangle) >= self.minimumSignSize:
                    print("Triangle: " + str(calculateArea(triangle)))
                    center = getCenter(triangle)
                    centers.append(center)
                    triangles.append([cnt])
        trianglesObj = {
            "info": triangles,
            "debugImg": gray,
            "coordText": centers,
            "text": "T"
        }
        self.detected["t"] = trianglesObj
        return contours, gray, triangles

    def detectRectangles(self, color):
        img = self.img
        img = removeAllButOneColor(img,color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 1)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                rectangle = [(approx[0][0][0],approx[0][0][1]),
                            (approx[1][0][0],approx[1][0][1]),
                            (approx[2][0][0],approx[2][0][1]),
                            (approx[3][0][0],approx[3][0][1])]
                if calculateArea(rectangle) >= self.minimumSignSize:
                    print("Rectangle: " + str(calculateArea(rectangle)))
                    center = getCenter(rectangle)
                    centers.append(center)
                    rectangles.append([cnt])
        rectanglesObj = {
            "info": rectangles,
            "debugImg": gray,
            "coordText": centers,
            "text": "r"
        }
        self.detected["r"] = rectanglesObj
        return contours, gray, rectangles

    def detectStop(self):
        img = self.img
        img = removeAllButOneColor(img,"red")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 1)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        stops = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 8:
                stop = [(approx[0][0][0],approx[0][0][1]),
                        (approx[1][0][0],approx[1][0][1]),
                        (approx[2][0][0],approx[2][0][1]),
                        (approx[3][0][0],approx[3][0][1]),
                        (approx[4][0][0],approx[4][0][1]),
                        (approx[5][0][0],approx[5][0][1]),
                        (approx[6][0][0],approx[6][0][1]),
                        (approx[7][0][0],approx[7][0][1])]
                if calculateArea(stop) >= self.minimumSignSize:
                    print("STOP: " + str(calculateArea(stop)))
                    center = getCenter(stop)
                    centers.append(center)
                    stops.append([cnt])
        stopsObj = {
            "info": stops,
            "debugImg": gray,
            "coordText": centers,
            "text": "STOP"
        }
        self.detected["STOP"] = stopsObj
        return contours, gray, stops