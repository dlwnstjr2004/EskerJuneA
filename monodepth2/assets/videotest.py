import cv2
cap = cv2.VideoCapture('Road.mp4')
while(cap.isOpened()):
    r, frame = cap.read()
    dst = frame.copy()
    dst = frame[360:720, 320: 960]
    cv2.imshow("src",frame)
    cv2.imshow("dst",dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

