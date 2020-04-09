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
    
    """
    Detect all the circles of the image
    """
    def detectCircles(self, color):
        circlesObj = self.getDefaultObj(color + " Circle")
        self.detected["c-" + color] = circlesObj
        for img in self.arrays[color]:
            self.detectCirclesInOneImage(color, img)
    """
    Detect all the triangles of the image
    """ 
    def detectTriangles(self, color):
        trianglesObj = self.getDefaultObj(color + " Triangle")    
        self.detected["t"] = trianglesObj
        for img in self.arrays[color]:
            self.detectTrianglesInOneImage(color, img)
    """
    Detect all the rectangles of the image
    """
    def detectRectangles(self, color):
        rectanglesObj = self.getDefaultObj(color + " Rectangle")
        self.detected["r-" + color] = rectanglesObj
        for img in self.arrays[color]:
            self.detectRectanglesInOneImage(color,img)
    """
    Detect all the stops in the image
    """ 
    def detectStop(self):
        stopsObj = self.getDefaultObj("STOP")
        self.detected["STOP"] = stopsObj
        for img in self.arrays["Red"]:
            self.detectStopInOneImage(img)

    def __init__(self, img, arrayImg):
        self.arrays = arrayImg
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.minimumSignSize = MINIMUM_SIGN_SIZE * h * w
        print("Minimum Sign Size: " + str(self.minimumSignSize))
        self.detected = {}
        self.detectedSigns = []

    def getDetected(self):
        return self.detected

    def getDetectedSigns(self):
        return self.detectedSigns

    """
    Prepare the image for detection
    """
    def prepareImg(self, color, img):
        img = removeAllButOneColor(img,color)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
  
    """
    Construct and return a default obj
    """ 
    def getDefaultObj(self, text):
        return {
            "info": [],
            "coordText": [],
            "text": text
        }
    
    """
    Detect all the circles of a color in one image
    """  
    def detectCirclesInOneImage(self, color, img):
        contours = self.prepareImg(color, img)
        circles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) > 8:
                circle = []
                for coords in approx:
                    circle.append((coords[0][0],coords[0][1]))
                if calculateArea(circle) >= self.minimumSignSize:
                    center = getCenter(circle)
                    centers.append(center)
                    circles.append([cnt])
                    self.detectedSigns.append({
                        "name": self.detected["c-" + color]["text"],
                        "sign": circle
                    })
        self.detected["c-" + color]["info"].append(circles)
        self.detected["c-" + color]["coordText"].append(centers)
        return contours, circles
    
    """
    Detect all the triangles of a color in one image
    """  
    def detectTrianglesInOneImage(self, color, img):
        contours = self.prepareImg(color, img)
        triangles = []
        centers = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                triangle = [(approx[0][0][0],approx[0][0][1]),
                            (approx[1][0][0],approx[1][0][1]),
                            (approx[2][0][0],approx[2][0][1])]
                if calculateArea(triangle) >= self.minimumSignSize:
                    center = getCenter(triangle)
                    centers.append(center)
                    triangles.append([cnt])
                    self.detectedSigns.append({
                        "name": self.detected["t"]["text"],
                        "sign": triangle
                    })
        self.detected["t"]["info"].append(triangles)
        self.detected["t"]["coordText"].append(centers)
        return contours, triangles

    """
    Detect all the rectangles of a color in one image
    """
    def detectRectanglesInOneImage(self, color, img):
        contours = self.prepareImg(color, img)
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
                    center = getCenter(rectangle)
                    centers.append(center)
                    rectangles.append([cnt])
                    self.detectedSigns.append({
                        "name": self.detected["r-" + color]["text"],
                        "sign": rectangle
                    })
        self.detected["r-" + color]["info"].append(rectangles)
        self.detected["r-" + color]["coordText"].append(centers)
        return contours, rectangles

    """
    Detect all the stops in one image
    """ 
    def detectStopInOneImage(self, img):
        contours = self.prepareImg("Red", img)    
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
                    center = getCenter(stop)
                    centers.append(center)
                    stops.append([cnt])
                    self.detectedSigns.append({
                        "name": self.detected["STOP"]["text"],
                        "sign": stop
                    })
        self.detected["STOP"]["info"].append(stops)
        self.detected["STOP"]["coordText"].append(centers)
        return contours, stops
