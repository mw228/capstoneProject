import cv2
import math
import numpy as np

#cascades that are used in this project
side_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")
smile_cascade =  cv2.CascadeClassifier("haarcascade_smile.xml")
glass_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

#video settings which involves getting name of video file and extracting the values for the fps resolution and the like
videoToUse = "Video Set 10 Camera 2.mp4"
vid =cv2.VideoCapture(videoToUse)
outputVideo = 'output - ' + videoToUse
video = cv2.VideoCapture(videoToUse)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
settings = cv2.VideoWriter_fourcc(*'MP4V')
outputVideo = cv2.VideoWriter(outputVideo, settings, fps, (int(width), int(height)), True)



#variable for x, y, w, h, frames since last seen, been seen this frame?
previousFaces = [[None,None,None, None, None ,False]]
previousEyes = [[None,None,None, None, None ,False]]
scaleFactorFaces = 0.2
#counter for how many frames have pasted. used for a few things including letting the user know how far along the program is. Also used in determining how to include the first face
frameCounter = 0
faceFrameCounter = 0
eyeFrameCounter = 0
mouthFrameCounter = 0



#checks if the video is still open
while vid.isOpened():
    #prints out the current frame and how many seconds of video have been completed
    print("Frame currently: " + str(frameCounter) + " and seconds are: " + str(int(frameCounter / fps)))
    #takes out the next frame in the video. ret is true or false whether the frame exists
    ret,frame = vid.read()
    #checks if ret if not it closes the files and exists the program
    if(ret):
        # convert the frame to grayscale
        grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #detect faces using two face classifiers, scaling the image down a bit, determing how strict the classfier is, setting maximium and minimum size of faces,
        faces = face_cascade.detectMultiScale(grayImg, scaleFactor = 1.2, minNeighbors=7,minSize=[100,100], maxSize = [500,500])
        sideFaces = side_cascade.detectMultiScale(grayImg, scaleFactor =1.2, minNeighbors=7, minSize=[100,100], maxSize = [500,500])
        #get the locatoin for each face found using 1 classifier
        for (x, y, w, h) in faces:
            #if its the first frame
            if((frameCounter == 0) and (not (previousFaces[0][0] == None))):
                #add it into the face array
                previousFaces[0] = [int(x),int(y),int(w), int(h),0,True]
            #face_detected is used to determine if there is overlap between the two face classifiers
            face_detected = True
            #this simply cycles though the other array and checks
            for (x2, y2, w2, h2) in sideFaces:
                if x2 > x + w or x2 + w2 < x or y2 > y + h or y2 + h2 < y:
                    continue  # no overlap
                else:
                    face_detected = False
                    break  # only draw one bounding box around each face
            #if a face has been found and there is no overlap then continue
            if face_detected:
                #newFace is used to determine if that face is not in the array of previous faces
                newFace = True
                #if there are no faces in it and if the first one is not None then there are faces already in the array so continue
                if((len(previousFaces) != 0) and (previousFaces[0][0] != None)):
                    #go through each face
                    for i in range(0,len(previousFaces),1):
                        #check if its within a scaled region of the previous faces
                        if((int(x) in range(int(previousFaces[i][0] * (1-scaleFactorFaces)), int(previousFaces[i][0] * (1+scaleFactorFaces)))) and (int(y) in range(int(previousFaces[i][1] * (1-scaleFactorFaces)), int(previousFaces[i][1] * (1+scaleFactorFaces))))):
                            #if it is add in that its been seen and reset the counter of frames since last seen. then say that it is not a new face
                            previousFaces[i][5] = True
                            previousFaces[i][4] = 0
                            newFace = False
                #if it is a new face then add it to the array
                if(newFace):
                    previousFaces.append([int(x),int(y), int(w), int(h), 0, True])
                #draw the face onto the frame
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                faceFrameCounter = faceFrameCounter + 1
                #create a smaller region of interest surrounding the face both in grayscale and color
                roi_gray = grayImg[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                #select smaller region of face to be used as eye regions and mouth regions of interest
                eye_roi = roi_color[int(0.2*h):int(0.7*h), int(0.2*w):int(0.8*w)]
                mouth_roi = roi_color[int(0.6*h):int(0.9*h), int(0.3*w):int(0.7*w)]

                #use the classifiers to determine if therea are mouths or eyes in their respective ROIS
                eyes = eye_cascade.detectMultiScale(eye_roi, 1.1, 2)
                smile = smile_cascade.detectMultiScale(mouth_roi, 1.15, 5)
                mouth = mouth_cascade.detectMultiScale(mouth_roi,1.15,4)
                mouthRegion = mouth
                #mouthRegion is by default mouth
                if(len(mouth)==0):
                    mouthRegion = smile
                if(len(smile)==0):
                    mouthRegion = mouth
                #cycle though all the eyes
                for (ex, ey, ew, eh) in eyes:
                    #draw a box around them using the values already provided
                    cv2.rectangle(roi_color, (int(0.2*w)+ex, int(0.2*h)+ey), 
                                (int(0.2*w)+ex+ew, int(0.2*h)+ey+eh), (255, 0, 0), 5)
                    eyeFrameCounter = eyeFrameCounter + 1
                #cycle though all mouths
                for (mx, my, mw, mh) in mouthRegion:
                    #draw a box around them using the values already provided
                    cv2.rectangle(roi_color, (int(0.3*w)+mx, int(0.6*h)+my), 
                                (int(0.3*w)+mx+mw, int(0.6*h)+my+mh), (0, 0, 255), 5)
                    mouthFrameCounter = mouthFrameCounter + 1
        #cycle though the other section of faces which are detected from the side
        for (x, y, w, h) in sideFaces:
            #if its the first frame and no faces are put in yet
            if((frameCounter == 0) and (not (previousFaces[0][0] == None))):
                #add it into the face array
                previousFaces[0] = [int(x),int(y),int(w), int(h),0,True]
            #face_detected is used to determine if there is overlap between the two face classifiers
            face_detected = True
            #this simply cycles though the other array and checks
            for (x2, y2, w2, h2) in faces:
                if x2 > x + w or x2 + w2 < x or y2 > y + h or y2 + h2 < y:
                    continue  # no overlap
                else:
                    face_detected = False
                    break  # only draw one bounding box around each face
            #if a face has been found and there is no overlap then continue
            if face_detected:
                #newFace is used to determine if that face is not in the array of previous faces
                newFace = True
                #if there are no faces in it and if the first one is not None then there are faces already in the array so continue
                if((len(previousFaces) != 0) and (previousFaces[0][0] != None)):
                    #go through faces
                    for i in range(0,len(previousFaces),1):
                        #check if its within a scaled region of previous faces
                        if((int(x) in range(int(previousFaces[i][0] * (1-scaleFactorFaces)), int(previousFaces[i][0] * (1+scaleFactorFaces)))) and (int(y) in range(int(previousFaces[i][1] * (1-scaleFactorFaces)), int(previousFaces[i][1] * (1+scaleFactorFaces))))):
                            #if it is add in that its been seen and reset the counter of frames since last seen. then say that it is not a new face
                            previousFaces[i][5] = True
                            previousFaces[i][4] = 0
                            newFace = False
                #if it is a new face then add it to the array
                if(newFace):
                    previousFaces.append([int(x),int(y), int(w), int(h), 0,True])
                #draw face to frame
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                faceFrameCounter = faceFrameCounter + 1
                #creat smaller ROIs like earlier centered around face
                roi_color_side = frame[y:y+h, x:x+w]
                roi_gray_side = grayImg[y:y+h, x:x+w]
                #select smaller region of face to be used as eye regions and mouth regions of interest
                eye_roi_side = roi_color_side[int(0.2*h):int(0.7*h), 0:int(0.9*w)]
                mouth_roi_side = roi_color_side[int(0.6*h):int(0.9*h), 0:int(0.5*w)]
                #use the classifiers to determine if therea are mouths or eyes in their respective ROIS
                eyes_side = eye_cascade.detectMultiScale(eye_roi_side, 1.1, 2)
                smile_side = smile_cascade.detectMultiScale(mouth_roi_side, 1.10, 4)
                mouth_side = mouth_cascade.detectMultiScale(mouth_roi_side,1.15,5)
                #change the mouth and use the mouth as defualt if there are none use the side
                mouthRegion_side = mouth_side
                if(len(mouth_side)==0):
                    mouthRegion_side = smile_side
                if(len(smile_side)==0):
                    mouthRegion_side = mouth_side
                # Draw rectangles around detected eyes and mouth
                for (ex, ey, ew, eh) in eyes_side:
                    #draw a box around them using the values already provided
                    cv2.rectangle(roi_color_side, (int(0.2*w)+ex, int(0.2*h)+ey), 
                                (int(0.2*w)+ex+ew, int(0.2*h)+ey+eh), (255, 0, 0), 5)
                    eyeFrameCounter = eyeFrameCounter + 1
                for (mx, my, mw, mh) in mouthRegion_side:
                    #draw a box around them using the values already provided
                    cv2.rectangle(roi_color_side, (mx, int(0.6*h)+my), 
                                (mx+mw, int(0.6*h)+my+mh), (0, 0, 255), 5)
                    mouthFrameCounter = mouthFrameCounter + 1
        #create a list for the indexes to be removed from the faces
        removeList = []
        #check if there are any faces to even check
        if((len(previousFaces) != 0) and (previousFaces[0][0] != None)):
            #go through each face
            for i in range(0, len(previousFaces), 1):
                #check if its been seen this frame
                if(not (previousFaces[i][5])):
                    #check how many frames it has been since its last been seen if its less than 10 frames
                    if(previousFaces[i][4] < 10):
                        #add 1 to how many frames have gone past
                        previousFaces[i][4] = previousFaces[i][4] + 1
                        #draw box for the face
                        frame = cv2.rectangle(frame, (previousFaces[i][0], previousFaces[i][1]), (previousFaces[i][0]+previousFaces[i][2], previousFaces[i][1]+previousFaces[i][3]), (0, 255, 0), 2)
                        faceFrameCounter = faceFrameCounter + 1
                    else:
                        #if its been more than 10 frames then schedule it to be removed
                        removeList.append(i)
                #if it has been seen
                else:
                    #reset it to false to prep for the next frame
                    previousFaces[i][5] = False
            #cycle though the remove list
            for x in range(0, len(removeList),1):
                #remove the faces not seen in a while
                previousFaces.pop(removeList[x])
        #add one to frame and write frame to output file
        cv2.putText(frame, "Frame: " + str(frameCounter), (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        cv2.putText(frame, "FaceFrame: " + str(faceFrameCounter), (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        cv2.putText(frame, "MouthFrame: " + str(mouthFrameCounter), (20,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        cv2.putText(frame, "EyeFrame: " + str(eyeFrameCounter), (20,80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        frameCounter = frameCounter + 1

        outputVideo.write(frame)
    else:
        #close videos
        vid.release()
        outputVideo.release()
#remove any windows that may be opened
cv2.destroyAllWindows()