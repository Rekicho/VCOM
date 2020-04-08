import cv2
import numpy as np

# Local imports
from utils import *

# Percentage of image size signs have to be to be considered
MINIMUM_SIGN_SIZE = 0.001 

class Detector:
    """ Detetor runs shape detection algorithms to find signs in the image and saves it in 'detected'

    This class can detect:
        - cicles:     detectCircles(color)
        - triangles:  detectTriangles(color)
        - rectangles: detectRectangles(color)
        - stop:       detectStop()
    Attributes:
        img (image): Image where the detection algorithms will be run 
        detected (:obj: type of signal -> information for the signal): The information of the image will be kept in this data structure and can then be exported for further use 
    """

    def __init__(self, img, arrayImg):
        self.arrays = arrayImg
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.minimumSignSize = MINIMUM_SIGN_SIZE * h * w
        print("Minimum Sign Size: " + str(self.minimumSignSize))
        self.detected = {}

    def getDetected(self):
        return self.detected

    def prepareImg(self, color, img):
        img = removeAllButOneColor(img,color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 1)
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
        return gray, ret, thresh, contours, h

    def detectCircles(self, color):
        circlesObj = {
            "info": [],
            "debugImg": None,
            "coordText": [],
            "text": "c-" + color
        }
        self.detected["c-" + color] = circlesObj
        for img in self.arrays[color]:
            self.detectCirclesInOneImage(color, img)

    def detectCirclesInOneImage(self, color, img):
        gray, ret, thresh, contours, h = self.prepareImg(color, img)
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
        
        print("circle: " + str(circles))
        print("centers: " + str(centers))
        self.detected["c-" + color]["info"].append(circles)
        self.detected["c-" + color]["coordText"].append(centers)
        return contours, gray, circles

    def detectTriangles(self, color):
        trianglesObj = {
            "info": [],
            "debugImg": None,
            "coordText": [],
            "text": "T"
        }
        self.detected["t"] = trianglesObj
        for img in self.arrays[color]:
            self.detectTrianglesInOneImage(color, img)

    def detectTrianglesInOneImage(self, color, img):
        gray, ret, thresh, contours, h = self.prepareImg(color, img)
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

        self.detected["t"]["info"].append(triangles)
        self.detected["t"]["coordText"].append(centers)
        return contours, gray, triangles

    def detectRectangles(self, color):
        rectanglesObj = {
            "info": [],
            "debugImg": None,
            "coordText": [],
            "text": "r"
        }
        self.detected["r"] = rectanglesObj
        for img in self.arrays[color]:
            self.detectRectanglesInOneImage(color,img)
    
    def detectRectanglesInOneImage(self, color, img):
        gray, ret, thresh, contours, h = self.prepareImg(color, img)
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

        print("rect: " + str(rectangles))
        print("centers: " + str(centers))
        self.detected["r"]["info"].append(rectangles)
        self.detected["r"]["coordText"].append(centers)
        return contours, gray, rectangles

    def detectStop(self):
        stopsObj = {
            "info": [],
            "debugImg": None,
            "coordText": [],
            "text": "STOP"
        }
        self.detected["STOP"] = stopsObj
        for img in self.arrays["red"]:
            self.detectStopInOneImage(img)


    def detectStopInOneImage(self, img):
        gray, ret, thresh, contours, h = self.prepareImg("red", img)    
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
        self.detected["STOP"]["info"].append(stops)
        self.detected["STOP"]["coordText"].append(centers)
        return contours, gray, stops
