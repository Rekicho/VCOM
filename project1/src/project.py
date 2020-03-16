import cv2
import sys
import numpy as np
import pytesseract
from pytesseract import Output

from signDetection import *
from utils import *

def acquireImage():
    if len(sys.argv) > 1: 
        img = openImage(sys.argv[1])
    else:
        img = takeImageFromCamera()
    return img

def drawCircles(img, circles):
    if not(circles is None):
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return img


img = acquireImage()
circles, img = detectCircles(img, 'red')
# img = detectTriangles(img)
# img = detectRectangles(img)
# img = detectStop(img)
# img = detectStopText(img)
img = drawCircles(img, circles)
cv2.imshow("Signs",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.png',img)