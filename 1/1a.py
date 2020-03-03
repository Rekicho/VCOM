import cv2

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    height = img.shape[0]
    width = img.shape[1]
    print("Height: " + str(height))
    print("Width: " + str(width))
    cv2.imshow(fileName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()