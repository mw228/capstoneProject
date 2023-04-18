import cv2, time
import numpy as np

hand_cascsade = cv2.CascadeClassifier("aGest.xml")
vid =cv2.VideoCapture("handTest.mp4")
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
        hand_roi = frame[y:y+h, x:x+w]
        # colorHand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_YCrCb2RGB)
        # grayHand_roi = cv2.cvtColor(colorHand_roi, cv2.COLOR_RGB2GRAY)

        # hands = hand_cascsade.detectMultiScale(hand_roi,1.4,3, minSize=[100,100])

        # for (hx, hy, hw, hh) in hands:
        #     cv2.rectangle(frame, (x+hx, y+hy), (x+hx+hw, y+hy+hh), (0, 255, 0), 2)


  
    
    
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