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


def calculate_position(bbox):
    pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1) # channel, rows, default cols
    dst = cv2.perspectiveTransform(pos, perspective_transform).reshape(-1, 1) # default channel, rows, defalut cols
    #print(pos)
    #print(dst)
    return np.array((UNWARPED_SIZE[1]-dst[1])/pixels_per_meter[1])

def Cutting_image(img, x1, y1, x2, y2):
    car_img = img[y1:y2, x1:x2, :]
    return car_img

if __name__ == '__main__':
    
    with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)
    
    perspective_transform = perspective_data["perspective_transform"]
    pixels_per_meter = perspective_data['pixels_per_meter']
    orig_points = perspective_data["orig_points"]

    #print(perspective_transform)
    
    cap = cv2.VideoCapture('Test.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        bbox = [100,100,300,550]
        #print("bbox : "+str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]))
        print(calculate_position(bbox))
        newimg = Cutting_image(frame,300,300,500,500)
        cv2.imshow('result',newimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
