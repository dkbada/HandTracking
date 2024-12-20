import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
pTime = 0.0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Fingers"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

#print(len(overlayList)) # checking if we imported all images

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb (right hand)
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)   
        print(totalFingers) 
    
        h,w,c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1] # height and width range

        cv2.putText(img, str(totalFingers), (30,300), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0), 10)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (480, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)