# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2 # opencv 사용
import numpy as np

def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=2): # 선 그리기
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, a=1, b=1., c=0.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    #print 'size = [%f, %f]\n' % (width, height)

    cv2.namedWindow('CAM_Window')
    cv2.resizeWindow('CAM_Window', 1280, 720)

    ret, image = cap.read()

    height, width = image.shape[:2] # 이미지 높이, 너비

    gray_img = grayscale(image) # 흑백이미지로 변환
    
    blur_img = gaussian_blur(gray_img, 3) # Blur 효과
        
    canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘

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

    cv2.imshow('result',hough_img) # Canny 이미지 출력
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break; 

cap.release()
cv2.destroyAllWindows()
