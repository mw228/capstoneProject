import cv2
import math
import numpy as np

side_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")
smile_cascade =  cv2.CascadeClassifier("haarcascade_smile.xml")
glass_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mySide = "VideoSet8Camera1.mp4"
theirSide = "VideoSet8Camera2.mp4"
vid =cv2.VideoCapture(mySide)
# vid = cv2.VideoCapture(0)

ret,frame = vid.read()
#variable for x, y, w, h, frames since last seen, been seen this frame?
previousFaces = [[None,None,None, None, None ,False]]
previousEyes = [[None,None,None, None, None ,False]]
scaleFactorFaces = 0.2
frameCounter = 0




while True:
    
# while True:
    ret,frame = vid.read()
    grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, scaleFactor = 1.2, minNeighbors=7,minSize=[100,100], maxSize = [500,500])
    sideFaces = side_cascade.detectMultiScale(grayImg, scaleFactor =1.2, minNeighbors=7, minSize=[100,100], maxSize = [500,500])
    # faceGlass = glass_cascade.detectMultiScale(grayImg, scaleFactor=1.2, minNeighbors=7, minSize=[100,100], maxSize=[500,500]) 

    # faces = []
    # for (x, y, w, h) in faces:
    #     faces.append((x, y, w, h))
    # for (x, y, w, h) in sideFaces:
    #     faces.append((x, y, w, h))
    # for (x, y, w, h) in faceGlass:
    #     faces.append((x, y, w, h))

    # # Remove duplicates
    # faces = list(set(faces))

    # # Draw the bounding boxes
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        if((frameCounter == 0)):
            previousFaces[0] = [int(x),int(y),int(w), int(h),0,True]
        face_detected = True
        for (x2, y2, w2, h2) in sideFaces:
            if x2 > x + w or x2 + w2 < x or y2 > y + h or y2 + h2 < y:
                continue  # no overlap
            else:
                face_detected = False
                break  # only draw one bounding box around each face
        if face_detected:
            newFace = True
            for i in range(0,len(previousFaces),1):
                if((int(x) in range(int(previousFaces[i][0] * (1-scaleFactorFaces)), int(previousFaces[i][0] * (1+scaleFactorFaces)))) and (int(y) in range(int(previousFaces[i][1] * (1-scaleFactorFaces)), int(previousFaces[i][1] * (1+scaleFactorFaces))))):
                    previousFaces[i][5] = True
                    previousFaces[i][4] = 0
                    newFace = False
            if(newFace):
                previousFaces.append([int(x),int(y), int(w), int(h), 0,True])
            
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = grayImg[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eye_roi = roi_color[int(0.2*h):int(0.7*h), int(0.2*w):int(0.8*w)]
            
            eyes = eye_cascade.detectMultiScale(eye_roi, 1.1, 2)
            
            # Define a new ROI for mouth detection
            mouth_roi = roi_color[int(0.6*h):int(0.9*h), int(0.3*w):int(0.7*w)]
            smile = smile_cascade.detectMultiScale(mouth_roi, 1.15, 5)
            mouth = mouth_cascade.detectMultiScale(mouth_roi,1.15,4)
            mouthRegion = mouth
            #mouthRegion is by default mouth
            if(len(mouth)==0):
                mouthRegion = smile
            if(len(smile)==0):
                mouthRegion = mouth
            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(roi_color, (int(0.2*w)+ex, int(0.2*h)+ey), 
                            (int(0.2*w)+ex+ew, int(0.2*h)+ey+eh), (255, 0, 0), 5)
            for (mx, my, mw, mh) in mouthRegion:
                cv2.rectangle(roi_color, (int(0.3*w)+mx, int(0.6*h)+my), 
                            (int(0.3*w)+mx+mw, int(0.6*h)+my+mh), (0, 0, 255), 5)
            # eyes = eye_cascade.detectMultiScale(roi_color, 1.2,2)
            # smile = smile_cascade.detectMultiScale(roi_color,1.7,9)
            # for (eex,eey,eew,eeh) in smile:
            #     cv2.rectangle(roi_color,(eex,eey),(eex+eew, eey+eeh), (0,0,255),5)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
            cv2.imshow("eyes Front on",eye_roi)
            cv2.waitKey(1)
            cv2.imshow("Face Front on",roi_color)
            cv2.waitKey(1)
            cv2.imshow("Mouth Front on",mouth_roi)
            cv2.waitKey(1)
    for (x, y, w, h) in sideFaces:
        if((frameCounter == 0) and (not (previousFaces[0][0] == None))):
            previousFaces[0] = [int(x),int(y),int(w), int(h),0,True]
        face_detected = True
        for (x2, y2, w2, h2) in faces:
            if x2 > x + w or x2 + w2 < x or y2 > y + h or y2 + h2 < y:
                continue  # no overlap
            else:
                face_detected = False
                break  # only draw one bounding box around each face
        if face_detected:
            newFace = True
            for i in range(0,len(previousFaces),1):
                if((int(x) in range(int(previousFaces[i][0] * (1-scaleFactorFaces)), int(previousFaces[i][0] * (1+scaleFactorFaces)))) and (int(y) in range(int(previousFaces[i][1] * (1-scaleFactorFaces)), int(previousFaces[i][1] * (1+scaleFactorFaces))))):
                    previousFaces[i][5] = True
                    previousFaces[i][4] = 0
                    newFace = False
            if(newFace):
                previousFaces.append([int(x),int(y), int(w), int(h), 0,True])
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_color_side = frame[y:y+h, x:x+w]
            roi_gray_side = grayImg[y:y+h, x:x+w]
            #define new eye ROI
            eye_roi_side = roi_color_side[int(0.2*h):int(0.7*h), 0:int(0.9*w)]
            eyes_side = eye_cascade.detectMultiScale(eye_roi_side, 1.1, 2)
            
            # Define a new ROI for mouth detection
            mouth_roi_side = roi_color_side[int(0.6*h):int(0.9*h), 0:int(0.5*w)]
            
            smile_side = smile_cascade.detectMultiScale(mouth_roi_side, 1.10, 4)
            mouth_side = mouth_cascade.detectMultiScale(mouth_roi_side,1.15,5)
            mouthRegion_side = mouth_side
            if(len(mouth_side)==0):
                mouthRegion_side = smile_side
            if(len(smile_side)==0):
                mouthRegion_side = mouth_side
            # Draw rectangles around detected eyes and mouth
            for (ex, ey, ew, eh) in eyes_side:
                cv2.rectangle(roi_color_side, (int(0.2*w)+ex, int(0.2*h)+ey), 
                            (int(0.2*w)+ex+ew, int(0.2*h)+ey+eh), (255, 0, 0), 5)
            for (mx, my, mw, mh) in mouthRegion_side:
                cv2.rectangle(roi_color_side, (mx, int(0.6*h)+my), 
                            (mx+mw, int(0.6*h)+my+mh), (0, 0, 255), 5)
            
            cv2.imshow("Eyes side",eye_roi_side)
            cv2.waitKey(1)
            cv2.imshow("Face Side",roi_color_side)
            cv2.waitKey(1)
            cv2.imshow("Mouth Side",mouth_roi_side)
            cv2.waitKey(1)

            # eyes = eye_cascade.detectMultiScale(roi_color, 1.2,2)
            # smile = smile_cascade.detectMultiScale(roi_color,1.7,9)
            # for (eex,eey,eew,eeh) in smile:
            #     cv2.rectangle(roi_color,(eex,eey),(eex+eew, eey+eeh), (0,0,255),5)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
    # for(x,y,w,h) in sideFaces:
    frameCounter = frameCounter + 1
    print(str(previousFaces))
    removeList = []
    for i in range(0, len(previousFaces), 1):
        if(not (previousFaces[i][5])):
            if(previousFaces[i][4] < 10):
                previousFaces[i][4] = previousFaces[i][4] + 1
                frame = cv2.rectangle(frame, (previousFaces[i][0], previousFaces[i][1]), (previousFaces[i][0]+previousFaces[i][2], previousFaces[i][1]+previousFaces[i][3]), (0, 255, 0), 2)
            else:
                removeList.append(i)
        else:
            previousFaces[i][5] = False
    for x in range(0, len(removeList),1):
        previousFaces.pop(removeList[x])
    
    
    # for x, y, w, h in faces:
    #     frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)
    #     roi_gray = grayImg[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_color, 1.2,2)
    #     smile = smile_cascade.detectMultiScale(roi_color,1.6,8)
    #     for (eex,eey,eew,eeh) in smile:
    #         img = cv2.rectangle(roi_color,(eex,eey),(eex+eew, eey+eeh), (0,0,255),5)
    #     for (ex,ey,ew,eh) in eyes:
    #         img = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),5)
    #         eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
    #         ret, grayThresh = cv2.threshold(eye_gray, 220,255,cv2.THRESH_BINARY)
    #         contours, hierarchy = cv2.findContours(grayThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         for contour in contours:
    #             area = cv2.contourArea(contour)
    #             rect= cv2.boundingRect(contour)
    #             x, y, width, height = rect
    #             radius = .25 * (width + height)

    #             area_condition = (100 <= area <= 200)
    #             symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.2)
    #             fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)

    #             if area_condition and symmetry_condition and fill_condition:
    #                 cv2.circle(roi_color, (int(x + radius), int(y + radius)), int(1.3*radius), (30,0,60), -1)

   
        
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