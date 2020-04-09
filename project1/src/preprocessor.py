import cv2
import numpy as np

from utils import *

class Preprocessor:
    """ This class preprocesses the image by removing noise and simplifying the colors
    In a first step, the image is processed sign by sign with the intent to fill the sign making it simpler and clean some noise in the image.
    In a second step, the signs are already well defined. There will be a dilute/erode pair that will fill the signs as much as possible. 
        The objective of this step is to interpret partially ocluded signs since they will gain shape.

    Attributes:
        img (image): Image that is going to get processed 
    """

    """
    Returns the results obtained from the processing
    """
    def getLists(self):
        return {
            "blue": self.blueList,
            "red": self.redList
        }

    """
    Returns the cleaned image
    """
    def getProcessed(self):
        return self.processedImg

    """
    Deletes the little noise in the image
    """
    def cleanImage(self, img):
        kernel1 = np.ones((5,5),np.uint8)
        img = cv2.dilate(img,kernel1,iterations = 1)
        img = cv2.erode(img,kernel1,iterations = 1)
        kernel2 = np.ones((5,5),np.uint8)
        img = cv2.erode(img,kernel2,iterations = 1)
        img = cv2.dilate(img,kernel2,iterations = 1)
        return img 

    """
    Finds all the signs and executes the processing in the signs
    """
    def processSingleSign(self, img, val, x , y):
        h = img.shape[0]
        w = img.shape[1]
        temp = img.copy()
        cv2.floodFill(temp, None, (x,y),(0,255,255))
        temp = removeAllButOneColor(temp, "yellow")
        x_offset = y_offset = val                   
        frame = np.zeros([h + y_offset*2, w + x_offset*2,3],dtype=np.uint8)
        frame[y_offset:y_offset+temp.shape[0], x_offset:x_offset+temp.shape[1]] = temp
        kernel = np.ones((val,val),np.uint8)
        frame = cv2.dilate(frame,kernel,iterations = 1)
        frame = cv2.erode(frame,kernel,iterations = 1)
        return frame[y_offset:y_offset+h, x_offset:x_offset+w]

    """
    Separates each sign according to its color and if the image needs to be cleaned or not
    An image needing to be cleaned means there is noise in the image that needs to be taken into account
    If an image does not need to be cleaned, then the erode/dilute will be bigger in order to fill the images
    """
    def processElements(self, color, clean = True):
        if clean:
            startImg = self.img
        else:
            startImg = self.processedImg
        img = removeAllButOneColor(startImg, color)
        everySign = []
        h = img.shape[0]
        w = img.shape[1]
        if clean:
            img = self.cleanImage(img)
            val = 100 # Smaller value to deal with the noise
        else:
            val = 300 # Bigger value to fill the signs properly
        
        # Look for signs
        for y in range(0, h):
            for x in range(0, w):
                if img[y][x][0] == RBG_PURE_COLOR[color][0] and img[y][x][1] == RBG_PURE_COLOR[color][1] and img[y][x][2] == RBG_PURE_COLOR[color][2]:
                    singleSign = self.processSingleSign(img.copy(), val, x, y)
                    everySign.append(singleSign)
                    cv2.floodFill(img, None, (x,y),(0,0,0))
        # Save computed results 
        self.saveResuts(clean, img, everySign, color)
        
    """
    Saves results in everySign according to the 'clean' and 'color' settings
    """
    def saveResuts(self, clean, img, everySign, color):
        h = img.shape[0]
        w = img.shape[1]
        finalImgHSV = np.zeros([h,w,3], dtype=np.uint8)
        if clean:
            for singleSign in everySign:
                img = convertToHSV(singleSign)
                yellow_mask = create_mask(img, ["yellow"])
                cv2.bitwise_or(finalImgHSV, img, finalImgHSV, mask=yellow_mask)

            finalImg = convertToRGB(finalImgHSV)
            finalImg[np.where((finalImg==RBG_PURE_COLOR["yellow"]).all(axis=2))] = RBG_PURE_COLOR[color]
            if color == "red":
                self.redProcessed = finalImg.copy()
            else:
                self.blueProcessed = finalImg.copy()
        else:
            for i in range(len(everySign)):
                everySign[i][np.where((everySign[i]==RBG_PURE_COLOR["yellow"]).all(axis=2))] = RBG_PURE_COLOR[color]
            if color == "red":
                self.redList = everySign
            else:
                self.blueList = everySign
        
    """
    Constructs and starts the preprocessing
    """
    def __init__(self, img):
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.blueList = []
        self.redList = []
        print("[PREPROCESSING] Cleaning the image")
        self.processElements('red')
        self.processElements('blue')
        temp = cv2.bitwise_or(self.blueProcessed, self.redProcessed)
        self.processedImg = temp.copy()
        print("[PREPROCESSING] Increasing quality of signs")
        self.processElements('red', False)
        self.processElements('blue', False)

    """
    Displays the images in the array for debugging purposes
    """
    def printArray(self, array):
        for pic in array:
            self.print2(pic)

    """
    Displays an image for debugging purposes
    """
    def print2(self, img):
        cv2.imshow("Signs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()