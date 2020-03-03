import cv2

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    roi = cv2.selectROI(img)
    crop = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.imwrite("eu_" + "crop.jpg", crop)