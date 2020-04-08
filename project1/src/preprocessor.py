import cv2
import numpy as np

from utils import *

class Preprocessor:
    """ This class preprocesses the image by removing noise and simplifying the colors    

    Attributes:
        img (image): Image that is going to get processed 
    """
    def print2(self, img):
        cv2.imshow("Signs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __init__(self, img):
        self.img = img
        h = img.shape[0]
        w = img.shape[1]
        self.blueList = []
        self.redList = []
        self.isolateEachElementOfColor('red')
        self.isolateEachElementOfColor('blue')
        temp = cv2.bitwise_or(self.blueProcessed, self.redProcessed)
        self.processedImg = temp.copy()
        self.isolateEachElementOfColor('red', False)
        self.isolateEachElementOfColor('blue', False)
        self.printArray(self.blueList)
        self.printArray(self.redList)
    
    def getLists(self):
        return {
            "blue": self.blueList,
            "red": self.redList
        }

    def getProcessed(self):
        return self.processedImg

    def printArray(self, array):
        for pic in array:
            self.print2(pic)

    def isolateEachElementOfColor(self, color, connect = True):
        if connect:
            startImg = self.img
        else:
            startImg = self.processedImg
        img = removeAllButOneColor(startImg, color)
        everySign = []
        h = img.shape[0]
        w = img.shape[1]
        if connect:
            kernel1 = np.ones((5,5),np.uint8)
            img = cv2.dilate(img,kernel1,iterations = 1)
            img = cv2.erode(img,kernel1,iterations = 1)
            kernel2 = np.ones((5,5),np.uint8)
            img = cv2.erode(img,kernel2,iterations = 1)
            img = cv2.dilate(img,kernel2,iterations = 1)
        if connect:
            val = 100
        else:
            val = 300
        for y in range(0, h):
            for x in range(0, w):
                if img[y][x][0] == RBG_PURE_COLOR[color][0] and img[y][x][1] == RBG_PURE_COLOR[color][1] and img[y][x][2] == RBG_PURE_COLOR[color][2]:
                    temp = img.copy()
                    #if not connect:
                    #    self.print2(temp)
                    cv2.floodFill(temp, None, (x,y),(0,255,255))
                    temp = removeAllButOneColor(temp, "yellow")
                    #if not connect:
                    #    self.print2(temp)
                    x_offset = y_offset = val                   
                    frame = np.zeros([h + y_offset*2, w + x_offset*2,3],dtype=np.uint8)
                    frame[y_offset:y_offset+temp.shape[0], x_offset:x_offset+temp.shape[1]] = temp
                    kernel = np.ones((val,val),np.uint8)
                    frame = cv2.dilate(frame,kernel,iterations = 1)
                    frame = cv2.erode(frame,kernel,iterations = 1)
                    singleSign = frame[y_offset:y_offset+h, x_offset:x_offset+w]
                    everySign.append(singleSign)
                    cv2.floodFill(img, None, (x,y),(0,0,0))
        finalImgHSV = np.zeros([h,w,3], dtype=np.uint8)
        if connect:
            for singleSign in everySign:
                img = convertToHSV(singleSign)
                yellow_mask = create_mask(img, ["yellow"])
                cv2.bitwise_or(finalImgHSV, img, finalImgHSV, mask=yellow_mask)

            finalImg = convertToRGB(finalImgHSV)
            finalImg[np.where((finalImg==RBG_PURE_COLOR["yellow"]).all(axis=2))] = RBG_PURE_COLOR[color]
            if color == "red":
                self.redProcessed = finalImg.copy()
                # self.print2(self.redProcessed)
            else:
                self.blueProcessed = finalImg.copy()
            
        else:
            for i in range(len(everySign)):
                everySign[i][np.where((everySign[i]==RBG_PURE_COLOR["yellow"]).all(axis=2))] = RBG_PURE_COLOR[color]
            if color == "red":
                self.redList = everySign
            else:
                self.blueList = everySign
        
