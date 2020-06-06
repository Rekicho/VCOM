import cv2 as cv
import csv
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

orb = cv.ORB_create()

def loadFeatures(datasetPath):
    descriptors = []
    with open(datasetPath + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img = cv.imread(datasetPath + '/' + row[0] + '.jpg')
            gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray,None)
            if des is None:
                continue
            descriptors.append((des, row[1]))
    return descriptors

def generateImFeatures(dataset):
    descriptors = dataset[0][0]
    for descriptor,malign in dataset[1:]:
        descriptors = np.vstack((descriptors, descriptor))  

    descriptors_float = descriptors.astype(float) 

    k = 200
    voc, variance = kmeans(descriptors_float, k, 1) 

    im_features = np.zeros((len(dataset), k), "float32")
    for i in range(len(dataset)):
        words, distance = vq(dataset[i][0],voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(dataset)+1) / (1.0*nbr_occurences + 1)), 'float32')

    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return im_features

def generateModel(dataset):
    for descriptor,malign in dataset:
        maligns.append(malign)

    im_features = generateImFeatures(dataset)

    clf = LinearSVC(max_iter=1000000)
    clf.fit(im_features, maligns)

    return clf
    

train = loadFeatures('dataORB/train')
test = loadFeatures('dataORB/test')
maligns = []

clf = generateModel(train)
im_features = generateImFeatures(test)

result = clf.predict(im_features)

hit = 0
miss = 0

for descriptor,malign in test:
    maligns.append(malign)

for i in range(len(result)):
    if result[i] == maligns[i]:
        hit+=1
    else:
        miss+=1

print(result)
print(hit)
print(miss)
print(str(hit * 100 / (hit+miss)) + "%")    