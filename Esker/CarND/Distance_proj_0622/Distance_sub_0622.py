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

import rospy
from std_msgs.msg import String
import serial
ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)
ser.flushInput()

def laser_listener():
    rospy.init_node('laser_listener', anonymous=True)
    sub = rospy.Subscriber('/DISTANCE', String, call2)
    rospy.spin()

def call2(data):
    #print(data.data)
    distance = float(data.data)
    #print(distance)
    if (distance >= 10):
        ser.write("1".encode())
        print("accelerate")
    elif ( (distance < 10) and (distance > 0) ):
        ser.write("0".encode())
        print("slow down")
    else:
        print("somethings bad happen")
        
if __name__ == '__main__':
    laser_listener()
    
