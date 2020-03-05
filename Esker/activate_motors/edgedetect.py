# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # �ѱ� �ּ������� �̰� �ؾ���
import cv2 # opencv ���
import numpy as np

def grayscale(img): # ����̹����� ��ȯ
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny �˰���
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # ����þ� ����
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    print 'size = [%f, %f]\n' % (width, height)

    cv2.namedWindow('CAM_Window')
    cv2.resizeWindow('CAM_Window', 1280, 720)

    ret, image = cap.read()

    height, width = image.shape[:2] # �̹��� ����, �ʺ�

    gray_img = grayscale(image) # ����̹����� ��ȯ
    
    blur_img = gaussian_blur(gray_img, 3) # Blur ȿ��
        
    canny_img = canny(blur_img, 70, 210) # Canny edge �˰���

    cv2.imshow('result',canny_img) # Canny �̹��� ���
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break; 

cap.release()
cv2.destroyAllWindows()
