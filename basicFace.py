import cv2, time
face_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
img= cv2.imread("testImg.jpg",1)
video = cv2.VideoCapture(0)
a = 1
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# while True:
#     a=a+1
#     # check,frame = video.read()
#     # print(frame)
#     # check,frame = img.read()
#     # print(frame)
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Capture",gray_img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# print(a)
# video.release()
# cv2.destroyAllWindows()

# print(check)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
print(type(faces))
print(faces)
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),3)
    resizedImg = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
    cv2.imshow("Test",img)
    # cv2.imshow("Resized",resizedImg)

cv2.waitKey(5000)
cv2.destroyAllWindows()
