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

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    #print 'size = [%f, %f]\n' % (width, height)

    cv2.namedWindow('CAM_Window')
    cv2.resizeWindow('CAM_Window', 1280, 720)

    ret, image = cap.read()

    height, width = image.shape[:2] # �̹��� ����, �ʺ�

    gray_img = grayscale(image) # ����̹����� ��ȯ
    
    blur_img = gaussian_blur(gray_img, 3) # Blur ȿ��
        
    canny_img = canny(blur_img, 70, 210) # Canny edge �˰���

    # Perform hough transform
    # Get first candidates for real lane lines  
    line_arr = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 10, 20)
    ##line_arr all zero. -> line_arr = line
    #draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)
    # Get slope degree to separate 2 group (+ slope , - slope)
    slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree)<160]
    slope_degree = slope_degree[np.abs(slope_degree)<160]
    # ignore vertical slope lines
    line_arr = line_arr[np.abs(slope_degree)>95]
    slope_degree = slope_degree[np.abs(slope_degree)>95]
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    #print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)
    
    # interpolation & collecting points for RANSAC
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    draw_circle(img,L_interp,(255,255,0))
    draw_circle(img,R_interp,(0,255,255))

    # erase outliers based on best line
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    cv2.imshow('result',hough_img) # Canny �̹��� ���
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break; 

cap.release()
cv2.destroyAllWindows()
