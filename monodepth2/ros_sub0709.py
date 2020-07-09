import math
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time

import rospy
from std_msgs.msg import String
import serial
ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)
ser.flushInput()

def laser_listener():
    rospy.init_node('laser_listener', anonymous=True)
    sub = rospy.Subscriber('/DEPTHS', String, call1)
    sub2 = rospy.Subscriber('/DEPTHS2',String, call2)
    rospy.spin()

def call1(data):
    #print(data.data)
    pixelsize = int(data.data)
    
    #ser.write(distance.encode())
    #print(pixelsize)
    if (pixelsize >= 400):
        ser.write("0".encode())
        print("stop")
    elif ( (distance < 400) and (distance >= 333) ):
        ser.write("1".encode())
        print("speed level 1")
    elif ( (distance < 333) and (distance >= 267) ):
        ser.write("2".encode())
        print("speed level 2")
    elif ( (distance < 267) and (distance >= 200) ):
        ser.write("3".encode())
        print("speed level 3")
    elif ( (distance < 200) and (distance >= 133) ):
        ser.write("4".encode())
        print("speed level 4")
    elif ( (distance < 133) and (distance >= 67) ):
        ser.write("5".encode())
        print("speed level 5")
    elif (distance < 67):
        ser.write("6".encode())
        print("fullspeed")
    else:
        print("somethings bad happen")
    ser.write("a".encode()) #a for accelerate
        
def call2(data):
    defalut_middle = 320 #we use 640x320 image so x(row) center is 320
    center = int(data.data)
    
    ser.write(str(defalut_middle-center).encode())
    ser.write("s".encode()) #s for steering
    '''
    if (defalut_middle>center):
        ser.write("r")
    elif (defalut_middle<center):
        ser.write("l")
    else:
        ser.write("s")
    '''

if __name__ == '__main__':
    laser_listener()
    
