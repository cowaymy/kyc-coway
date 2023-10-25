import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box

import cv2
import numpy as np
import pandas as pd

import util
import uuid
#from sort.sort import *
from util import read_license_plate, df_to_json ,df_to_cvs
from PIL import Image
import json
from collections import OrderedDict

import face_recognition


import matplotlib.pyplot as plt

from deepface import DeepFace
#Read more at: https://viso.ai/computer-vision/deepface/

results = {}
json_data = OrderedDict()

# mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
coway_model = YOLO('./models/coway_nric_bast_v0.1.pt')

res = coway_model('./data/images/10.jpeg')

df = pd.DataFrame( columns =['tran_id','idx' ,'type','value', 'bbox_conf','value_conf','cropFaceImg'])

def noise_removal(image):
    import numpy as np 
    kernel =np.ones((1,1) , np.uint8)
    image =cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((0,0) , np.uint8)  #글씨굵어짐 
    image =cv2.erode(image, kernel, iterations=1)
    image =cv2.morphologyEx(image,cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image,3)
    return (image)


rIdx =0
for r in res:
	trin_id  =uuid.uuid4()
	results = {}
	
	arr = r.plot()
	image = Image.fromarray(arr[..., ::-1])
	r.save_crop('datasets/yolo/images/crop' ,file_name=f'{trin_id}.jpg')
	
	cropImagName ="datasets/yolo/images/crop/IMAGE/"f'{trin_id}'".jpg"
	


	
	for  indx, d in  enumerate(r.boxes):
		if r.names[int(d.cls)] != 'IMAGE':
		
			image = save_one_box(d.xyxy, r.orig_img.copy(), BGR=True, save=False)
			# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			# image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
			# image = noise_removal(image)
			license_plate_text, license_plate_text_score = read_license_plate(image)
			# image = Image.fromarray(image)
			# image.show()
			
			df.loc[rIdx]={'tran_id': str(trin_id),
				'idx':indx ,
				'type':r.names[int(d.cls)] ,
				'value':license_plate_text ,
				'bbox_conf' :float(d.conf), 
				'value_conf' :license_plate_text_score,
				'cropFaceImg':f'{str(trin_id)}.jpg'}
			
			rIdx = rIdx+1
			# process license plate
	    
			# license_plate_crop_gray = cv2.cvtColor(r.orig_img.copy(), cv2.COLOR_BGR2GRAY)
			# _,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
			# license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
			# lic
			# ense_plate_crop_gray.show()
			#
			#results__[0][r.names[int(d.cls)] ] ={}
	
		else:
			image = save_one_box(d.xyxy, r.orig_img.copy(), BGR=True, save=False)
			image = Image.fromarray(image)

			face = face_recognition.face_encodings(np.array(image))[0]
			realface = face_recognition.face_encodings(face_recognition.load_image_file(r".\datasets\real\capture\images\real\realFace.jpg"))[0]
			results = face_recognition.compare_faces([face], realface, tolerance=0.45)
			print(results)

			fig,axs= plt.subplots(1,2 , figsize=(15,5))
			axs[0].imshow(plt.imread(cropImagName))
			axs[1].imshow(plt.imread(r'datasets\real\capture\images\real\realFace.jpg'))
		
			fig.suptitle(f" Verifie :: { results }")
			plt.show()


	
			# models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
			# result = DeepFace.verify(img1_path =cropImagName, img2_path = r"D:\OCR\automatic-number-plate-recognition-python-yolov8\datasets\real\capture\images\real\test.jpg" ,model_name=models[3])
			
			# fig,axs= plt.subplots(1,2 , figsize=(15,5))
			# axs[0].imshow(plt.imread(cropImagName))
			# axs[1].imshow(plt.imread(r'D:\OCR\automatic-number-plate-recognition-python-yolov8\datasets\real\capture\images\real\test.jpg'))


			# fig.suptitle(f" Verifie :: { result['verified']} - Distance ::  { result['distance']:0.4}  ")
			# plt.show()




print(df)
#df_to_json(df)
df_to_cvs(df)
