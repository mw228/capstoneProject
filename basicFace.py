import cv2, time, face_recognition
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade =  cv2.CascadeClassifier("haarcascade_smile.xml")
# img= cv2.imread("faceImg.png",1)
vid=cv2.VideoCapture("testVideo.mp4")
# video = cv2.VideoCapture(0)
face_locations=[]

a = 1
gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

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

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=10)

# print(type(faces))
# print(faces)
while True:
    ret,frame = vid.read()
    for (x,y,w,h) in faces:
        face_gray = gray_img[y:y+h, x:x+w]
        face_color = vid[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_img, 1.2,5)
    img=cv2.rectangle(vid,(x,y), (x+w,y+h), (0,255,0),3)
#     # resizedImg = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
    for (ex,ey,ew,eh) in eyes:
        img = cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)
        smile = smile_cascade.detectMultiScale(vid,1.7,23)
# # print(smile)
    for (eex,eey,eew,eeh) in smile:
        img = cv2.rectangle(img,(eex,eey),(eex+eew, eey+eeh), (0,255,0),5)
    cv2.imshow("Test",vid)
    # cv2.imshow("Resized",resizedImg)

# cv2.waitKey(30000)
# cv2.destroyAllWindows()
