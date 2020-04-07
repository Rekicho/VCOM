import cv2
import sys
import numpy as np
import pytesseract
from pytesseract import Output

# Local imports
from utils import *
from detector import Detector
from printer import Printer
from reader import Reader
from preprocessor import Preprocessor

# Read the image
reader = Reader()
img = reader.getImage()
original = img.copy()

# Preprocess the image
pre = Preprocessor(img)
img = pre.getProcessed()

# Detect image Points of Interest (POI)
det = Detector(img)
det.detectCircles("red")
det.detectCircles("blue")
det.detectTriangles("red")
det.detectRectangles("blue")
det.detectStop()
ans = det.getDetected()

# Print the Detected POI's into the image
printer = Printer(original)
original = printer.printAllIntoImage(ans)
printer.showAndSave()

