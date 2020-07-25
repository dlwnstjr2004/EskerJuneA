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

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI ����

    mask = np.zeros_like(img) # mask = img�� ���� ũ���� �� �̹���
    
    if len(img.shape) > 2: # Color �̹���(3ä��)��� :
        color = color3
    else: # ��� �̹���(1ä��)��� :
        color = color1
        
    # vertices�� ���� ����� �̷��� �ٰ����κ�(ROI �����κ�)�� color�� ä�� 
    cv2.fillPoly(mask, vertices, color)
    
    # �̹����� color�� ä���� ROI�� ��ħ
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # �� �׸���
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # ���� ��ȯ
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, a=1, b=1., c=0.): # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, a, img, b, c)

def curve_detect(xl, xr, xl_old, xr_old, threshold, far_point):

    flag = 0
    for i in range(0,340):
        flag = img2[height-far_point, (int)(width/2-i)]
        if (flag == 255):
            if ( abs(xl-xl_old) < threshold ):
                xl = (int)(width/2-i)
                xl_old = xl
                break
            else:
                xl = xl_old
    if (flag == 0):
        xl = xl_old
    
    flag = 0
    for i in range(0,340):
        flag = img2[height-far_point, (int)(width/2+i)]
        if (flag == 255):
            if ( abs(xr-xr_old) < threshold ):
                xr = (int)(width/2+i)
                xr_old = xr
                break
            else:
                xr = xr_old
    if (flag == 0):
        xr = xr_old
    
    center = (int)((xl+xr)/2)
    
    return xl, xr, xl_old, xr_old, center

cap = cv2.VideoCapture('assets/nevada.mp4') # ������ �ҷ�����

#init curve detect parameters
xl1, xl2, xr1, xr2, xl1_old, xl2_old, xr1_old, xr2_old, center1, center2, center, center_old  = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

while(cap.isOpened()):
    ret, img = cap.read()

    height, width = img.shape[:2] # �̹��� ����, �ʺ�
    '''
    vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices) # ROI ����
    hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) # ���� ��ȯ
    #result = weighted_img(hough_img, image) # ���� �̹����� ����� �� overlap
    '''
    mask_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    #mask_img2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    cv2.rectangle(mask_img, ( (int)(width/2-350), height-60 ), ( (int)(width/2+350), height ), (255,255,255), thickness = 60)
    #cv2.rectangle(mask_img2, ( (int)(width/2-350), height-60 ), ( (int)(width/2+350), height-40 ), (255,255,255), thickness = 35)
    dst_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8) 
    #cv2.bitwise_and(img, (mask_img+mask_img2), dst_img)
    cv2.bitwise_and(img, mask_img, dst_img)

    gray_img = grayscale(dst_img)
    blur_img = gaussian_blur(gray_img, 3)
    canny_img = canny(blur_img, 70, 210)
    
    img2 = canny_img

    #find two points of roads
    xl1, xr1, xl1_old, xr1_old, center1 = curve_detect(xl1,xr1,xl1_old,xr1_old, 20, 10)
    xl2, xr2, xl2_old, xr2_old, center2 = curve_detect(xl2,xr2,xl2_old,xr2_old, 20, 50)
    if ( abs(center-center_old) < 20 ):
        center = center2 - center1
    else:
        center = center_old
    print("center2-center1 : %d", center) #if nagative, go left. else go right
    
    cv2.imshow('result',canny_img) # ��� �̹��� ���
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release
cap.release()
cv2.destroyAllWindows()
