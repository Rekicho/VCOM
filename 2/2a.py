import cv2
import numpy as np

if __name__ == "__main__":
    img = np.zeros((100,200,3), np.uint8)
    img[::]=(100,100,100)
    cv2.line(img, (0, 0), (200, 100), (255,255,255), 1)
    cv2.line(img, (200, 0), (0, 100), (255,255,255), 1)
    cv2.imshow("2a",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()