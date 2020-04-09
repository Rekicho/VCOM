import cv2
import sys
import numpy as np

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
coloredSigns = pre.getLists()

# Detect image Points of Interest (POI)
det = Detector(img, coloredSigns)
det.detectCircles("Red")
det.detectCircles("Blue")
det.detectTriangles("Red")
det.detectRectangles("Blue")
det.detectStop()
ans = det.getDetected()

# Print the Detected POI's into the image
printer = Printer(original)
printer.printToSTDOUT(det.getDetectedSigns())
original = printer.printAllIntoImage(ans)
printer.showAndSave()

