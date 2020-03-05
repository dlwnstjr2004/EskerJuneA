import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)
if ser.isOpen():
    print "Port Open"
#print ser.write("START\n")
print ser.write("0000\n")
ser.close()
