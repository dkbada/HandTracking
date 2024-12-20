import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280, 720
pTime = 0.0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Headers"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[4]

detector = htm.handDetector()

color = (255, 255, 255)
brushThickness = 10
eraserThickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    # index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp() # left hand

        # Two fingers if select mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), color, cv2.FILLED)
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    color = (255, 0, 0)
                elif 550 < x1 < 700:
                    header = overlayList[1]
                    color = (0, 0, 255)
                elif 700 < x1 < 900:
                    header = overlayList[2]
                    color = (0, 255, 0)
                elif 950 < x1 < 1100:
                    header = overlayList[3]
                    color = (0, 0, 0)

        # Index finger if drawing mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, eraserThickness)

            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), color, brushThickness)
            
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV) # the drawn region will be black, the rest white
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting header image
    img[0:125, 0:1280] = header

    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("image", img)
    cv2.imshow("invimg", imgInv)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)