import cv2
# import face_recognition
# import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")
smile_cascade =  cv2.CascadeClassifier("haarcascade_smile.xml")
vid =cv2.VideoCapture("testVideo.mp4")
# vid = cv2.VideoCapture(0)
# face_locations=[]
ret,frame = vid.read()
while True:
    
# while True:
    ret,frame = vid.read()
    grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, scaleFactor = 1.25, minNeighbors=7)
    
    # cv2.imshow('video', frame)
    
#     # face_locations = face_recognition.face_locations(rgb_frame)
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
        roi_gray = grayImg[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color, 1.2,2)
        smile = smile_cascade.detectMultiScale(roi_color,1.6,8)

        for (ex,ey,ew,eh) in eyes:
            img = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
    # cv2.rectangle(frame,(0,0,255),2)
        for (eex,eey,eew,eeh) in smile:
            img = cv2.rectangle(roi_color,(eex,eey),(eex+eew, eey+eeh), (0,0,255),5)
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