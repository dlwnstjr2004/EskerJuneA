#!/opt/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import time

CAM_ID = 0

cam = cv2.VideoCapture(1) #카메라 생성
if cam.isOpened() == False: #카메라 생성 확인
    exit()

#카메라 이미지 해상도 얻기
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH);
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT);

#윈도우 생성 및 사이즈 변경
cv2.namedWindow('CAM_Window')
cv2.resizeWindow('CAM_Window', 1280, 720)

while(True):
    prev_time = time.time()
    
    #카메라에서 이미지 얻기
    ret, frame = cam.read()

    curr_time = time.time()
    exec_time = curr_time - prev_time
    
    info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1))
    cv2.putText(frame, text=info, org=(50, 70),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=(255, 0, 0), thickness=2)
    
    #얻어온 이미지 윈도우에 표시
    cv2.imshow('CAM_Window', frame)

    #10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
        break;
        
#윈도우 종려
cam.release()
cv2.destroyWindow('CAM_Window')
