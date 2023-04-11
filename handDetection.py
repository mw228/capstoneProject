import cv2, time

hand_cascsade = cv2.CascadeClassifier("aGest.xml")
vid =cv2.VideoCapture("testVideo.mp4")
# img = cv2.imread("handImg.png")
# ret,frame = vid.read()
# gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)




    # cv2.imshow("Hands",img)

# cv2.waitKey(10000)
# cv2.destroyAllWindows()
while True:
    
# while True:
    ret,frame = vid.read()
    grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hands = hand_cascsade.detectMultiScale(grayImg,1.01,9)
    
    # cv2.imshow('video', frame)
    
#     # face_locations = face_recognition.face_locations(rgb_frame)
    for(x,y,w,h) in hands:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
    cv2.imshow('Video',frame)
#     if cv2.waitKey(25) == 13:
#         break
#     # Video_Capture.release()
#     # vid.release()
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
#     vid.release()