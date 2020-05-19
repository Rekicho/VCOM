import cv2 as cv
import csv

def loadFeatures(datasetPath):
    descriptors = []
    with open(datasetPath + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = cv.imread(datasetPath + '/' + row[0] + '.jpg')
            descriptors.append((img, row[1]))
    return descriptors

# train = loadFeatures('data/train')
test = loadFeatures('data/test')
