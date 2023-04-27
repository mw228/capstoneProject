import cv2, time
import numpy as np

hand_cascsade = cv2.CascadeClassifier("aGest.xml")
videoToUse = "Video Set 10 Camera 3.mp4"
vid =cv2.VideoCapture(videoToUse)
outputVideo = 'output - ' + videoToUse
video = cv2.VideoCapture(videoToUse)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
settings = cv2.VideoWriter_fourcc(*'MP4V')
outputVideo = cv2.VideoWriter(outputVideo, settings, fps, (int(width), int(height)), True)
frameCounter = 0
handFrameCounter = 0
while vid.isOpened():
    print("Frame currently: " + str(frameCounter) + " and seconds are: " + str(int(frameCounter / fps)))
    ret,frame = vid.read()
    if(ret):
        grayImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Define range of skin color in YCrCb color space
        lower_skin = np.array([0, 135, 70], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)

        # Threshold the YCrCb image to get only skin color pixels
        mask = cv2.inRange(ycrcb_img, lower_skin, upper_skin)

        # Apply morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.erode(mask, kernel, iterations=2)

        # Find contours in the filtered image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around each contour
        for contour in contours:
            area= cv2.contourArea(contour)
            if area>150000 or area<10000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            handFrameCounter = handFrameCounter + 1
            hand_roi = frame[y:y+h, x:x+w]
        cv2.putText(frame, "Frame: " + str(frameCounter), (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        cv2.putText(frame, "FaceFrame: " + str(handFrameCounter), (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        frameCounter = frameCounter + 1

        outputVideo.write(frame)
    else:
        #close videos
        vid.release()
        outputVideo.release()
cv2.destroyAllWindows()