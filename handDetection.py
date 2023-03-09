import cv2, time

hand_cascsade = cv2.CascadeClassifier("aGest.xml")

img = cv2.imread("handImg.png")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

hands = hand_cascsade.detectMultiScale(gray_img,1.01,9)

for(x,y,w,h) in hands:
    img=cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),5)

    cv2.imshow("Hands",img)

cv2.waitKey(10000)
cv2.destroyAllWindows()