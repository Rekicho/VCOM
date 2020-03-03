import cv2

if __name__ == "__main__":
    fileName = input("Enter .jpg file name: ")
    img = cv2.imread(fileName + ".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(fileName,img)
    cv2.imshow(fileName + "_gray",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(fileName + "_gray.jpg", gray)