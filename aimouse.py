import cv2
import autopy
import time
import numpy as np
import pyHandTrackingModule as htm

whCam , hhCam = 640,480
frameR = 100
smoothening = 7

pTime = 0
plocx,plocy = 0,0
clocx,clocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,whCam)
cap.set(4,hhCam)
detector = htm.handDetector(maxHands=1)
wScr,hScr = autopy.screen.size()

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)

    if len(lmList)!= 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        cv2.rectangle(img,(frameR,frameR),(whCam-frameR,hhCam - frameR),(255,0,255),2)

        if fingers[1] == 1  and fingers[2] == 0: 
            x2,y2 = lmList[12][1:]
            x3 = np.interp(x1,(frameR,whCam - frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hhCam - frameR),(0,hScr))
            clocx = plocx +(x3 - plocx)/smoothening
            clocy = plocy +(y3 - plocy)/smoothening

            autopy.mouse.move(wScr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)

        if fingers[1] == 1  and fingers[2] == 0:
            length,img,lineInfo = detector.findDistance(8,12,img)
            print(length)
            if length < 39 :
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(255,0,255),cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(40,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)








