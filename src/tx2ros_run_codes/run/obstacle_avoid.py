#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time
import serial
ser = serial.Serial('/dev/ttyACM0',115200,timeout=1)
ser.flushInput()

from darkflow.net.build import TFNet

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
set_session(tf.Session(config=config))

option = {
    'model': 'cfg/yolov2.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.45,
    'gpu': 1.0
}

tfnet = TFNet(option)
capture = cv2.VideoCapture(1)
capture.set(3, 320)
capture.set(4, 160)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

def callback(data):
    vel_msg = Twist()
    frame = np.zeros((500, 500,3), np.uint8)
    angle = data.angle_max
    angle_min = data.angle_min
    Vx = 250
    Vy = 250
    for r in data.ranges[0:80]:
        if r == float ('Inf'):
#            r = data.range_max
            r = 0
        '''elif r == 'None':
            r = 0.0
        elif r < 0.1:
            r = 0.0 '''

        x = math.trunc( (r * 10)*math.sin(-angle))
        y = math.trunc( (r * 10)*math.cos(angle))

 
        cv2.line(frame,(250, 250),(x+250,y+250),(255,0,0),1)
        Vx+=x
        Vy+=y
        angle= angle - data.angle_increment

    for r in data.ranges[80:280]:
        angle= angle - data.angle_increment

    for r in data.ranges[280:360]:
        if r == float ('Inf'):
#            r = data.range_max
            r = 0

        x = math.trunc( (r * 10)*math.sin(-angle))
        y = math.trunc( (r * 10)*math.cos(angle))

 
        cv2.line(frame,(250, 250),(x+250,y+250),(255,0,0),1)
        Vx+=x
        Vy+=y
        angle= angle - data.angle_increment

    cv2.line(frame,(250, 250),(250+Vx,250+Vy),(0,0,255),3)
    cv2.circle(frame, (250, 250), 2, (255, 255, 0))
    ang = -(math.atan2(Vx,Vy)-3.1416)*180/3.1416

    cv2.imshow('frame',frame)
    cv2.waitKey(1)

    if (270 < ang < 360):
        degree = ang - 270
    elif (0 <= ang < 90):
        degree = ang + 90
    else:
        degree = 0


#    if (0 <= ang < 70):
#	print ('low_angle is {}', ang)
#        degree = 90 - ang
#    elif (290 < ang < 360):
#	print ('high_angle is {}', ang)
#        degree = 450 - ang
#    else:
#        degree = 0

    message = str(int(degree))
    ser.write(message.encode())
    ser.write(";".encode())


    print('angle: %5.3f' %degree)


def call2(data):
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            label = result['label']
            if label=='person':
                #ser.write("0".encode())
		#ser.write(",".encode())
                print("I see human")
            elif label=='stop sign':
                while 1:
                    print('im stoping ')
                    print("Good Job!")
                    ser.write('0'.encode())
                    ser.write(';'.encode())
                    time.sleep(1)
                    print("STOP ME!")
            else:
                #ser.write("4".encode())
		#ser.write(",".encode())
                print("I see somethin but not human")

    else:
        capture.release()
        cv2.destroyAllWindows()
    '''if obj_name == 'stop sign':
        print('im stoping ')
        rospy.signal_shutdown('Terminated............')'''

def laser_listener():
    rospy.init_node('laser_listener', anonymous=True)
    rsub = rospy.Subscriber('/scan', LaserScan, callback, queue_size=1)
    sub = rospy.Subscriber('/YOLO', String, call2)
    rospy.spin()

if __name__ == '__main__':
    laser_listener()
