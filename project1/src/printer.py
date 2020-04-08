import cv2
import sys
import numpy as np

# Local Imports
from utils import *

class Printer:
    """The Printer allows the user to print information on top of an image
    
    The user can 'printLabels' and 'printShapes' on the image

    Attributes:
        img (image): The image where the contents should be printed 
    """

    def __init__(self, img):
        self.img = img
    
    def printLabels(self, img, obj):
        # Constants
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        color = (-1, 255, 0)
        thickness = 1

        # Personalized for each shape
        textToPrint = obj["text"]
        coords = obj["coordText"]
        print(textToPrint)
        print("coords: " + str(coords))
        for coord in coords:
            # print(coord[0])
            if len(coord) > 0:
                img = cv2.putText(img, textToPrint, coord[0], font, fontScale, color, thickness, cv2.LINE_AA)
        return img

    def printShapes(self, img, obj):
        text = obj["text"]
        #print("==================================")
        for sign in obj["info"]:
            #print(sign)
            if len(sign) > 0:
                cv2.drawContours(img, sign[0], 0, (0, 255, 0), 6)
        return img

    def printAllIntoImage(self, answer):
        for signType in answer:
            obj = answer[signType]
            #print(obj)
            self.img = self.printShapes(self.img, obj)
            self.img = self.printLabels(self.img, obj)
        return self.img

    def showAndSave(self):
        cv2.imshow("Signs", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('output.png', self.img)
