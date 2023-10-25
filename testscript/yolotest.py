from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4.480)



coco_model = YOLO('yolov8n.pt')