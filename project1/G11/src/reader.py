import cv2
import sys
import numpy as np

FPS = 60

class Reader:
    
    """Reads an image from the user
    """

    def openImage(self, name):
        return cv2.imread(name,cv2.IMREAD_COLOR)

    def takeImageFromCamera(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            img = frame
            cv2.imshow('Camera',img)
            if cv2.waitKey(int(1000/FPS)) != -1:
                break
        cap.release()
        cv2.destroyAllWindows()
        return img

    def getImage(self):
        if len(sys.argv) > 1: 
            return self.openImage(sys.argv[1])
        else:
            return self.takeImageFromCamera()
