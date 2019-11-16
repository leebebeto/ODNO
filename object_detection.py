from google.colab import drive
import os 
from os.path import join
import matplotlib.pylab as plt
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
import sys
from google.colab.patches import cv2_imshow

root = '/content/drive/'
drive.mount(root)
mot = "My Drive/Colab Notebooks/DeepLearning/"   # a custom path. you can change if you want to
MOT_PATH = join(root,mot)
img_data = join(MOT_PATH,'train/MOT17-09/img1/')
sys.path.append(img_data)
txt_data_path = join(MOT_PATH,'train/MOT17-09/gt')
f = open(txt_data_path+'/gt.txt', 'rb')
txt_data = [ line for line in f.readlines()]


COCO_INSTANCE_CATEGORY_NAMES = [
'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def sample_img(img_path):
	list_img_data = os.listdir(img_data)  
	list_img_data.sort()
	img_ex_path = img_data + list_img_data[0]
	img_ex_origin = cv2.imread(img_ex_path)
	img_ex = cv2.cvtColor(img_ex_origin, cv2.COLOR_BGR2RGB)
	# plt.imshow(img_ex)
	# plt.axis('off')
	# plt.show()
	return img_ex_path

def get_prediction(img_path, threshold):
	img = Image.open(img_path) # Load the image
	transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
	img = transform(img) # Apply the transform to the image
	pred = model([img]) # Pass the image to the model

	pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] 
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
	pred_score = list(pred[0]['scores'].detach().numpy())
	pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
	pred_boxes = pred_boxes[:pred_t+1]
	pred_class = pred_class[:pred_t+1]
	return img, pred_boxes, pred_class

def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=1.5, text_th=3): 
	img, boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
	img = cv2.imread(img_path) # Read image with cv2
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
	for i in range(len(boxes)):
		cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
		cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
	plt.figure(figsize=(15,20)) # display the output image
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.show()
	return img, boxes, pred_cls

def redundant_blurry(img, boxes, pred_cls):
	box_size_dict = {}
	box_coord_dict = {}
	for i in range(len(boxes)):
		x_len = abs(boxes[i][0][0] - boxes[i][1][0])
		y_len = abs(boxes[i][0][1] - boxes[i][1][1])
		box_size = x_len * y_len
		box_size_dict[i] = box_size
		box_coord_dict[i] = boxes[i] 
	box_size_dict = sorted(box_size_dict.items(), key = lambda x: x[1], reverse = True)
	for i in range(len(box_size_dict)):
		box_idx = box_size_dict[i][0]
		if i ==0:		
			cv2.rectangle(img, box_coord_dict[box_idx][0], box_coord_dict[box_idx][1],color=(0, 255, 0), thickness= 3) # Draw Rectangle with the coordinates
		else:
			crop_img = img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])]
			noised_img = cv2.blur(crop_img,(7,7),0)
			img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])] = noised_img
			cv2.rectangle(img, box_coord_dict[box_size_dict[i][0]][0], box_coord_dict[box_size_dict[i][0]][1],color=(0, 255, 0), thickness= 3) # Draw Rectangle with the coordinates

	plt.figure(figsize=(15,20)) # display the output image
	plt.imshow(img)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
img_ex_path = sample_img(img_data)
img, boxes, pred_cls = object_detection_api(img_ex_path,threshold=0.8)
redundant_blurry(img, boxes, pred_cls)