import cv2
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()
polylinesAr = []

#Input
video_adress = "C:\\Users\\deadp\\Desktop\\Object_tracking\\los_angeles.mp4"
video_adress1 = "C:\\Users\\deadp\\Desktop\\Object_tracking\\berlin.mp4"
video_adress2 = "C:\\Users\\deadp\\Desktop\\Object_tracking\\highway.mp4"
cap = cv2.VideoCapture(video_adress)
cap.open(video_adress)
print(cap.isOpened())

#Detector
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

#Output
while True:
    ret, frame = cap.read()

    #Object detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        #Removing noise
        area = cv2.contourArea(cnt)
        if area > 125:
            #cv2.drawContours(frame, [cnt],  -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x,y,w,h])

    #Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id, x1, y1, x2, y2 = box_id
        pt = [x2, y2]
        pt2 = [x1, y1]
        line = np.array([pt, pt2])
        polylinesAr.append(line)
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (x, y - 7), 0, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for line in polylinesAr:
            pt1 = line[0]
            pt2 = line[1]
            cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 0, 0), thickness=2)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()