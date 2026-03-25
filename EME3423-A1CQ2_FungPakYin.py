import cv2
import numpy as np
import serial
import  time

ser = serial.Serial('COM6',baudrate=115200,timeout=1)
time.sleep(0.5)
pos = 90
pos1 = 90

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('HUELOW', 'Trackbars', 90, 255, nothing)
cv2.createTrackbar('HUEHIGH', 'Trackbars', 120, 255, nothing)
cv2.createTrackbar('SATLOW', 'Trackbars', 160, 255, nothing)
cv2.createTrackbar('SATHIGH', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('VALLOW', 'Trackbars', 85, 255, nothing)
cv2.createTrackbar('VALHIGH', 'Trackbars', 255, 255, nothing)

cam= cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    _,img = cam.read()
    cv2.imshow('ori', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    huelow = cv2.getTrackbarPos('HUELOW', 'Trackbars')
    huehigh = cv2.getTrackbarPos('HUEHIGH', 'Trackbars')
    satlow = cv2.getTrackbarPos('SATLOW', 'Trackbars')
    sathigh = cv2.getTrackbarPos('SATHIGH', 'Trackbars')
    vallow = cv2.getTrackbarPos('VALLOW', 'Trackbars')
    valhigh = cv2.getTrackbarPos('VALHIGH', 'Trackbars')

    FGmask = cv2.inRange(hsv,(huelow,satlow,vallow),(huehigh,sathigh,valhigh))
    cv2.imshow("FGmask", FGmask)

    contours, hierarchy = cv2.findContours(FGmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)


        if area >=100:
            if area >= biggest_contour:
               (x, y, w, h) = cv2.boundingRect(cnt)
               cv2.rectangle(img ,(x,y),(x+w,y+h),(255,0,0),3)

               errorPan = (x + w / 2) - 640 / 2

               if abs(errorPan) > 20:
                   pos = pos - errorPan / 100

               if pos >= 170:
                   pos = 170

               if pos <= 10:
                   pos = 10

               servoPos = 'RL' + str(pos) + '\r'
               ser.write(servoPos.encode('utf-8'))
               time.sleep(0.01)

               errorPan = (y + h / 2) - 480 / 2

               if abs(errorPan) > 20:
                   pos1 = pos1 - errorPan / 100

               if pos1 >= 170:
                   pos1 = 170

               if pos1 <= 10:
                   pos1 = 10

               servoPos = 'UD' + str(pos1) + '\r'
               ser.write(servoPos.encode('utf-8'))
               time.sleep(0.01)

    cv2.imshow('final',img)

    if cv2.waitKey(1) ==27:
        break
