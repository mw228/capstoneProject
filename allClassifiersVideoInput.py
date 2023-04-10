import cv2
import time
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade =  cv2.CascadeClassifier("haarcascade_smile.xml")

videoToUse = 'testV8C1.mp4'
outputVideo = 'output - ' + videoToUse

video = cv2.VideoCapture(videoToUse)

fps = int(video.get(cv2.CAP_PROP_FPS))
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)


forcc = cv2.VideoWriter_fourcc(*'MJPG')
ouput = cv2.VideoWriter(outputVideo, forcc, fps, (int(width), int(height)), True)

frameNumber = 0

while video.isOpened():
    print("Current Frame is: " + str(frameNumber))
    ret, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=10)
    for (x,y,w,h) in faces:
        face_gray = gray_img[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_img, 1.2,5)
        frame=cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0),3)
        for (ex,ey,ew,eh) in eyes:
            frame = cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)
        smile = smile_cascade.detectMultiScale(frame,1.7,23)
        for (eex,eey,eew,eeh) in smile:
            frame = cv2.rectangle(frame,(eex,eey),(eex+eew, eey+eeh), (0,255,0),5)
        ouput.write(frame)
    #cv2.imshow('frame', frame)
    #if cv2.waitKey(1) == ord('q'):
    #    break
    frameNumber = frameNumber + 1
video.release()
ouput.release()
cv2.destroyAllWindows()
