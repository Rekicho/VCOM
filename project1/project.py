import cv2
import sys
import numpy as np
import pytesseract
from pytesseract import Output

FPS = 60

def openImage(name):
    return cv2.imread(name,cv2.IMREAD_COLOR)

def takeImageFromCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(True):
        ret, frame = cap.read()
        img = frame
        cv2.imshow('Camera',img)
        if cv2.waitKey(int(1000/FPS)) != -1:
            break
    cap.release()
    cv2.destroyAllWindows()
    return img

def convertToHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR);

HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 100, 100]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([170, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([101, 39, 64]),
            'upper': np.array([140, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 25% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 26-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ]
}

def create_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """

    # noinspection PyUnresolvedReferences
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            # noinspection PyUnresolvedReferences
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask

# color - angle
def removeAllButOneColor(img):
    red_mask = create_mask(img, ['red'])

    mask_img = cv2.bitwise_and(img, img, mask=red_mask)
    return mask_img

def detectCircles(img):
    #r = img.copy()
    #r[:, :, 1] = 0
    #r[:, :, 0] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return img

def detectTriangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==3:
            cv2.drawContours(img,[cnt],0,(0,255,0),-1)

    return img

def detectRectangles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            cv2.drawContours(img,[cnt],0,(255,0,0),-1)

    return img

def detectStop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==8:
            cv2.drawContours(img,[cnt],0,(255,0,0),-1)

    return img

def detectStopText(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low=np.array([150])
    high=np.array([255])
    mask = cv2.inRange(gray, low, high)
    gray[mask > 0] = 0
    # print(pytesseract.image_to_string(gray, lang='eng', config='--psm 10 --oem 3'))
    d = pytesseract.image_to_data(gray, lang='eng', config='--psm 10 --oem 3', output_type=Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if d['text'][i] == 'STOP':
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

if len(sys.argv) > 1: 
    img = openImage(sys.argv[1])
    #img = detectTriangles(img)
    #img = detectRectangles(img)
    #img = detectCircles(img)
    # img = detectStop(img)
    img = detectStopText(img)
else:
    img = takeImageFromCamera()
cv2.imshow("Sign",img)
cv2.waitKey(0)
cv2.destroyAllWindows()