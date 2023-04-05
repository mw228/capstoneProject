import cv2
import face_recognition
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
vid =cv2.VideoCapture("testVideo.mp4")
# vid = cv2.VideoCapture(0)
face_locations=[]
ret,frame = vid.read()
while True:
    
# while True:
    ret,frame = vid.read()
    grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, scaleFactor = 1.05, minNeighbors=10)
    # cv2.imshow('video', frame)
    
#     # face_locations = face_recognition.face_locations(rgb_frame)
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
    # cv2.rectangle(frame,(0,0,255),2)
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