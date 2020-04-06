import cv2
import sys
import numpy as np
import pytesseract
from pytesseract import Output

# from signDetection import *
from utils import *

from detector import Detector
from printer import Printer
from reader import Reader
from preprocessor import Preprocessor

# Read the image
reader = Reader()
img = reader.getImage()
original = img.copy()

pre = Preprocessor(img)
img = pre.getProcessed()
# Detect Point of Interests (POI)
det = Detector(img)
det.detectCircles("red")
det.detectCircles("blue")
det.detectTriangles("red")
det.detectRectangles("blue")
det.detectStop()
# det.printProcess()
# det.process()
ans = det.getDetected()

# Print the Detected POI's into the image
printer = Printer(original)
original = printer.printAllIntoImage(ans)
printer.showAndSave()

# circles, _ = detectCircles(img, 'blue')
# img = detectTriangles(img)
# img = detectRectangles(img)
# img = detectStop(img)
# img = detectStopText(img)
# img = drawCircles(img, circlesInfo)
