import cv2
import numpy as np

FPS = 60

"""
Convert the image from RGB to HSV
"""
def convertToHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

"""
Convert the image from HSV to RGB
"""
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

# RBG pure colors for comparison
RBG_PURE_COLOR = {
    "Red": [0,0,255],
    "Blue": [255,0,0],
    "Yellow": [0, 255, 255],
    "White": [255,255,255],
    "Green": [0,255,0]
}

# HSV pure colors for comparison
HSV_PURE_COLOR = {
    "Red": [0,100,255],
    "Blue": [255,0,0],
    "Yellow": [0, 255, 255],
    "Green": [120,100,100]
}

# HSV ranges for some possible colors of signs
HSV_RANGES = {
    # red is a major color
    'Red': [
        {
            'lower': np.array([0, 200, 100]),
            'upper': np.array([5, 255, 255])
        },
        {
            'lower': np.array([175, 200, 100]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'Yellow': [
        {
            'lower': np.array([25, 200, 100]),
            'upper': np.array([35, 255, 255])
        }
    ],
    # green is a major color
    'Green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'Cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'Blue': [
        {
            'lower': np.array([100, 200, 64]),
            'upper': np.array([141, 255, 255])
        }
    ],
    # violet is a minor color
    'Violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 25% of V
    'Black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 26-89% of V
    'Gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'White':    [
        {
            'lower': np.array([0, 0, 150]),
            'upper': np.array([180, 38, 255])
        }
    ]
}

"""
Creates a binary mask from HSV image using given colors.
"""
def create_mask(hsv_img, colors):
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask

def print2(img):
        cv2.imshow("Signs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Removes all the colors except for the one given 
def removeAllButOneColor(img, color):
    img = convertToHSV(img)
    colored_mask = create_mask(img, [color])
    mask_img = cv2.bitwise_and(img, img, mask=colored_mask)
    mask_img = convertToRGB(mask_img)
    # print2(mask_img)
    h = mask_img.shape[0]
    w = mask_img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if mask_img[y][x][0] != 0 or mask_img[y][x][1] != 0 or mask_img[y][x][2] != 0:
                mask_img[y][x] = RBG_PURE_COLOR[color]
    return mask_img

# Finds center of shape
def getCenter(shape):
    center = [ int(sum(x) / len(shape)) for x in zip(*shape) ]
    return (center[0], center[1])

# Calculates the area of a shape
def calculateArea(shape):
    soma = 0
    for i in range(len(shape)):
        if i == len(shape) - 1:
            soma += (shape[i][0] * shape[0][1]) - (shape[i][1] * shape[0][0])
        else:
            soma += (shape[i][0] * shape[i+1][1]) - (shape[i][1] * shape[i+1][0])
    return abs(soma) / 2
