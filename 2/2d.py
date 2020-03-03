import cv2
import numpy as np
import random

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    row,col,ch = img.shape
    num_salt = np.ceil(0.1 * img.size * 0.5)
    noise = np.copy(img)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
    noise[coords] = 255
    num_pepper = np.ceil(0.1 * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
    noise[coords] = 0
    cv2.imshow("2d",noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()