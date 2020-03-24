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

    def print2(self, img):
        cv2.imshow("Signs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def isolateEachElementOfColor(self, color):
        img = removeAllButOneColor(self.img, color)
        everySign = []
        h = img.shape[0]
        w = img.shape[1]
        
        #self.print2(img)
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
                    kernel = np.ones((50,50),np.uint8)
                    frame = cv2.dilate(frame,kernel,iterations = 1)
                    frame = cv2.erode(frame,kernel,iterations = 1)

                    kernel = np.ones((10,10),np.uint8)
                    #self.print2(frame)
                    frame = cv2.erode(frame,kernel,iterations = 1)
                    #self.print2(frame)
                    frame = cv2.dilate(frame,kernel,iterations = 1)
                    # self.print2(frame)
                    singleSign = frame[y_offset:y_offset+h, x_offset:x_offset+w]
                    # self.print2(singleSign)
                    everySign.append(singleSign)
                    cv2.floodFill(img, None, (x,y),(0,0,0))
        finalImg = np.zeros([h,w,3], dtype=np.uint8)
        for singleSign in everySign:
            img = convertToHSV(singleSign)
            yellow_mask = create_mask(img, ["yellow"])
            cv2.bitwise_or(finalImg, img, finalImg, mask=yellow_mask)
            self.print2(convertToRGB(finalImg))
        # self.print2(img)


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
                "text": color
            }
            self.detected[color] = circlesObj
        return circles, gray, centers
