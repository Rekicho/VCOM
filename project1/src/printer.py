import cv2
import sys
import numpy as np
import pytesseract
from pytesseract import Output

from signDetection import *
from utils import *

def drawCircles(img, circles):
    if not(circles is None):
        for i in circles[0,:]:
            center = (i[0],i[1])
            # draw the outer circle
            cv2.circle(img,center,i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,center,2,(0,0,255),3)
    return img

def drawTriangles(img, triangles):
    for triangle in triangles:
        cv2.drawContours(img, triangle, 0, (0, 255, 0), -1)
    return img

def drawRectangles(img, rectangles):
    for rectangle in rectangles:
        cv2.drawContours(img, rectangles, 0, (255, 0, 0), -1)
    return img

    # Print on top of 'img' all the labels in 'obj'
def printLabels(img, obj):
    # Constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    # Personalized for each shape
    textToPrint = obj["text"]
    coords = obj["coordText"]
    for coord in coords:
        img = cv2.putText(img, textToPrint, coord, font, fontScale, color, thickness, cv2.LINE_AA)
    return img

def printShapes(img, obj):
    text = obj["text"]
    if text == "redC" or text == 'blueC':
        img = drawCircles(img, obj["info"])
    elif text == "T":
        img = drawTriangles(img, obj["info"])
    elif text == "R":
        img = drawRectangles(img, obj["info"])
    return img

class Printer:
    def __init__(self, img):
        self.img = img

    def printAllIntoImage(self, answer):
        for signType in answer:
            obj = answer[signType]
            self.img = printShapes(self.img, obj)
            self.img = printLabels(self.img, obj)
        return self.img

    def getImage(self, img):
        return self.img

    def showAndSave(self):
        cv2.imshow("Signs", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('output.png', self.img)


