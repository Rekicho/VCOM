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

    def isolateEachElementOfColor(self, color):
        img = removeAllButOneColor(self.img, color)
        self.print2(img)
        everySign = []
        h = img.shape[0]
        w = img.shape[1]

        kernel1 = np.ones((5,5),np.uint8)
        img = cv2.dilate(img,kernel1,iterations = 1)
        # self.print2(img)
        img = cv2.erode(img,kernel1,iterations = 1)
        kernel2 = np.ones((5,5),np.uint8)
        img = cv2.erode(img,kernel2,iterations = 1)
        img = cv2.dilate(img,kernel2,iterations = 1)
        self.print2(img)
        
        for y in range(0, h):
            for x in range(0, w):
                if img[y][x][0] == RBG_PURE_COLOR[color][0] and img[y][x][1] == RBG_PURE_COLOR[color][1] and img[y][x][2] == RBG_PURE_COLOR[color][2]:
                    temp = img.copy()
                    cv2.floodFill(temp, None, (x,y),(0,255,255))
                    temp = removeAllButOneColor(temp, "yellow")
                    # temp only has one sign at a time!!
                    x_offset = y_offset = 100                   
                    frame = np.zeros([h + y_offset*2, w + x_offset*2,3],dtype=np.uint8)
                    frame[y_offset:y_offset+temp.shape[0], x_offset:x_offset+temp.shape[1]] = temp
                    kernel = np.ones((100,100),np.uint8)
                    frame = cv2.dilate(frame,kernel,iterations = 1)
                    # self.print2(frame)
                    frame = cv2.erode(frame,kernel,iterations = 1)
                    # kernel = np.ones((10,10),np.uint8)
                    # frame = cv2.erode(frame,kernel,iterations = 1)
                    # frame = cv2.dilate(frame,kernel,iterations = 1)
                    singleSign = frame[y_offset:y_offset+h, x_offset:x_offset+w]
                    everySign.append(singleSign)
                    cv2.floodFill(img, None, (x,y),(0,0,0))
        finalImgHSV = np.zeros([h,w,3], dtype=np.uint8)
        for singleSign in everySign:
            img = convertToHSV(singleSign)
            yellow_mask = create_mask(img, ["yellow"])
            cv2.bitwise_or(finalImgHSV, img, finalImgHSV, mask=yellow_mask)

        myMask = create_mask(finalImgHSV, ["yellow"])
        finalImg = convertToRGB(finalImgHSV)
        finalImg[np.where((finalImg==RBG_PURE_COLOR["yellow"]).all(axis=2))] = RBG_PURE_COLOR[color]

        if color == "red":
            self.redProcessed = finalImg.copy()
        else:
            self.blueProcessed = finalImg.copy()

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

    def detectTriangles(self):
        img = self.redProcessed #self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 1)
        contours, h = cv2.findContours(thresh, 1, 2)
        triangles = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                print("found a triangle")
                triangles.append([cnt])
        trianglesObj = {
            "info": contours,
            "debugImg": thresh,
            "coordText": triangles,
            "text": "T"
        }
        # print(trianglesObj)
        self.detected["t"] = trianglesObj
        return contours, thresh, triangles

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