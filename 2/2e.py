import cv2
import numpy as np

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    red = img.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0
    green = img.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    blue = img.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    cv2.imshow("red",red)
    cv2.imshow("green",green)
    cv2.imshow("blue",blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(fileName + "_red.jpg", red)
    cv2.imwrite(fileName + "_green.jpg", green)
    cv2.imwrite(fileName + "_blue.jpg", blue)