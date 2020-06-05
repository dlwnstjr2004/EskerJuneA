import math
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog

from lane_finder import LaneFinder
from settings import CALIB_FILE_NAME, PERSPECTIVE_FILE_NAME, UNWARPED_SIZE, ORIGINAL_SIZE

if __name__ == '__main__':
    with open('classifier.p', 'rb') as f:
        data = pickle.load(f)

    scaler = data['scaler']
    cls = data['classifier']

    window_size=[64, 80, 96, 112, 128, 160]#, 192, 224, 256]
    window_roi=[((200, 400),(1080, 550)), ((100, 400),(1180, 550)), ((0, 380),(1280, 550)),
                ((0, 360),(1280, 550)), ((0, 360),(1280, 600)), ((0, 360),(1280, 670)) ]#,

    with open(CALIB_FILE_NAME, 'rb') as f:
        calib_data = pickle.load(f)
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        img_size = calib_data["img_size"]

    with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)

    perspective_transform = perspective_data["perspective_transform"]
    pixels_per_meter = perspective_data['pixels_per_meter']
    orig_points = perspective_data["orig_points"]



    def process_image(img, lane_finder, cam_matrix, dist_coeffs, reset = False):
        img = cv2.undistort(img, cam_matrix, dist_coeffs)
        lane_finder.find_lane(img, distorted=False, reset=reset)
        return lane_finder.draw_lane_masked(img)#lane_finder.draw_lane_weighted(img)

    cap = cv2.VideoCapture('Test.mp4')

    while(cap.isOpened()):

        prev_time = time.time() #frame check start

        ret, frame = cap.read()
        
        lf = LaneFinder(ORIGINAL_SIZE, UNWARPED_SIZE, cam_matrix, dist_coeffs,
                        perspective_transform, pixels_per_meter, "warning.png")

        res_img = process_image(frame, lf, cam_matrix, dist_coeffs, True)

        curr_time = time.time() #frame check finish
        exec_time = curr_time - prev_time

        info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1)) #insert information
        cv2.putText(res_img, text=info, org=(50, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2) # font init
        
        cv2.imshow('result',res_img-frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


