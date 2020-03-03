import cv2

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    cv2.imwrite(fileName + ".bmp", img)