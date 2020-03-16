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
            center = (i[0],i[1])
            # draw the outer circle
            cv2.circle(img,center,i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,center,2,(0,0,255),3)
    return img

# Prints the features into the original image
#   img - Original image
#   answer - Dictionary with all the detected information
def printFeatures(img, answer):
    for signType in answer:
        obj = answer[signType]
        img = printShapes(img, obj)
        img = printLabels(img, obj) 
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


    #for center in centers:
    #img = cv2.putText(img, 'Fernandes e gay', center, font, fontScale, color, thickness, cv2.LINE_AA)

# Print on top 'img' all the shapes in 'obj'
def printShapes(img, obj):
    text = obj["text"]
    if text == "RC" or text == 'BC':
        img = drawCircles(img, obj["info"])
    return img

# Create the answer dicionary
# For each type of sign, there should be:
#   Information to print the circles
#   Coordinates where to print the legend for each sign (these calculations may vary from sign to sign)
#   The image for debuging purposes
answer = {}

img = acquireImage()

circlesInfo, debug, centers = detectCircles(img, "red")
# print(circlesInfo)
if circlesInfo is not None:
    circlesObj = {
        "info": circlesInfo,
        "debugImg": debug,
        "coordText": centers,
        "text": "RC"
    }
    answer["redCircles"] = circlesObj

circlesInfo2, img2, centers2 = detectCircles(img, "blue")
if circlesInfo2 is not None:
    circlesObj2 = {
        "info": circlesInfo2,
        "debugImg": img2,
        "coordText": centers2,
        "text": "BC"
    }
    answer["blueCircles"] = circlesObj2




# Line thickness of 2 px 


# circles, _ = detectCircles(img, 'blue')
# img = detectTriangles(img)
# img = detectRectangles(img)
# img = detectStop(img)
# img = detectStopText(img)
# img = drawCircles(img, circlesInfo)
img = printFeatures(img, answer)
cv2.imshow("Signs",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.png',img)