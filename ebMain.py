
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
from util import checkBlink

# main
cap = cv2.VideoCapture(0)

count = 0
while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  if checkBlink(img_ori) :
     count += 1
     if count >= 5:
      break

  cv2.imshow('result', img_ori)
  if cv2.waitKey(1) == ord('q'):
    break