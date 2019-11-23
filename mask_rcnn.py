import torch
import torchvision
from torchvision import transforms as T
from torchvision import models
import sys
import os 
from os.path import join
import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image


def test():
	print("hello world")

def sample_img(img_path):
	list_img_data = os.listdir(img_data)  
	list_img_data.sort()
	img_ex_path = img_data + list_img_data[98]
	img_ex_origin = cv2.imread(img_ex_path)
	# img_ex = cv2.cvtColor(img_ex_origin, cv2.COLOR_BGR2RGB)
	# plt.imshow(img_ex_origin)
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
	img = cv2.imread(img_path)
	blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
	mask = cv2.imread("./mask.png")

	output = np.where(mask==np.array([255, 255, 255]), blurred_img, img)
	cv2.imwrite("./output.png", output)
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
	plt.imshow(img); plt.show()
 
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
    return rgb

def segment(net, device, path, show_orig=True):
  img = Image.open(path)
#   trf = T.Compose([T.Resize(640), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
  trf = T.Compose([T.ToTensor()])
  inp = trf(img).unsqueeze(0).to(device)
  inp2 = trf(img).squeeze().to(device).detach().cpu().numpy()
  print(inp2.shape)
  out = net.to(device)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  iteration = 0
  for i in range(om.shape[0]):
	  for j in range(om.shape[1]):
		  iteration +=1
		  if om[i,j] == 15:
			  inp2[:,i,j] = 256
  plt.imshow(img); plt.axis('off'); plt.show()
  rgb = decode_segmap(om)
  print(rgb.shape)
  inp2 = np.transpose(inp2, (1,2,0))
  print('inp2', inp2.shape)
  plt.imshow(inp2); plt.axis('off'); plt.show()

MOT_PATH = os.getcwd()
img_data = join(MOT_PATH,'train/MOT17-09/img1/')
print(img_data)

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
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
img_ex_path = sample_img(img_data)
img, boxes, pred_cls = object_detection_api(img_ex_path,threshold=0.8)
redundant_blurry(img, boxes, pred_cls)

# trf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# fcn = models.segmentation.fcn_resnet101(pretrained=True)
# img_ex_path = sample_img(img_data)
# img= Image.open(img_ex_path)
# plt.imshow(img); plt.show()
# segment(fcn, device, img_ex_path)