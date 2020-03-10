import cv2, sys

FPS = 60

def openImage(name):
    return cv2.imread(name)

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

if len(sys.argv) > 1: 
    img = openImage(sys.argv[1])
else:
    img = takeImageFromCamera()
cv2.imshow("Sign",img)
cv2.waitKey(0)
cv2.destroyAllWindows()