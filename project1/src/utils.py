import cv2
import numpy as np

FPS = 60



def convertToHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

RBG_PURE_COLOR = {
    "red": [0,0,255],
    "blue": [255,0,0],
    "yellow": [0, 255, 255]
}

HSV_PURE_COLOR = {
    "red": [0,100,255],
    "blue": [255,0,0],
    "yellow": [0, 255, 255]
}

HSV_RANGES = {
    # red is a major color
    'red': [
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
            'lower': np.array([100, 200, 64]),
            'upper': np.array([141, 255, 255])
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

# Removes all the colors except for the one given 
def removeAllButOneColor(img, color):
    img = convertToHSV(img)
    red_mask = create_mask(img, [color])
    mask_img = cv2.bitwise_and(img, img, mask=red_mask)
    mask_img = convertToRGB(mask_img)
    # print(mask_img[50][50])
    h = mask_img.shape[0]
    w = mask_img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if mask_img[y][x][0] != 0 or mask_img[y][x][1] != 0 or mask_img[y][x][2] != 0:
                mask_img[y][x] = RBG_PURE_COLOR[color]
    return mask_img