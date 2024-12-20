import cv2
import mediapipe as mp # type: ignore
import numpy as np
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL # type: ignore 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume # type: ignore

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0.0
cTime = 0.0
 
detector = htm.handDetector(detectionCon=0.7)

# pycaw volume control initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volume.SetMasterVolumeLevel(-65.25, None)
volRange = volume.GetVolumeRange() 
minVol = volRange[0]
maxVol = volRange[1]
volBar = np.array([400])
vol = np.array([0])
volPer = np.array([0])


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2 
        length = math.hypot(x2-x1, y2-y1)
        #print(length) # 30 - 180

        cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)   # thumb point
        cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)   # index point
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)   # line between thumb and index
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)   # mindpoint
        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        
        vol = np.interp(length, [20, 190], [minVol, maxVol])
        volBar = np.interp(length, [20, 190], [400, 150])
        volPer = np.interp(length, [20, 190], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
    cv2.putText(img, f'Volume %: {str(int(volPer))}', (10, 70), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)   
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 45), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)