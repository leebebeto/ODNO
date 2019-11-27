from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from functools import partial
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
import random

class Ui_Form(object):
	def setupUi(self, Form):
		Form.setObjectName("Form")
		Form.resize(1001, 753)
		self.verticalLayoutWidget = QtWidgets.QWidget(Form)
		self.verticalLayoutWidget.setGeometry(QtCore.QRect(139, 179, 861, 571))
		self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
		self.resultLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
		self.resultLayout.setContentsMargins(0, 0, 0, 0)
		self.resultLayout.setObjectName("resultLayout")

		self.upload_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
		self.upload_button.setMaximumSize(QtCore.QSize(300, 16777215))
		self.upload_button.setObjectName("upload_button")
		self.resultLayout.addWidget(self.upload_button, 0, QtCore.Qt.AlignHCenter)

		self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
		self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1001, 181))
		self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
		self.photo_video_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
		self.photo_video_layout.setContentsMargins(0, 0, 0, 0)
		self.photo_video_layout.setObjectName("photo_video_layout")
		self.photo_button = QtWidgets.QToolButton(self.horizontalLayoutWidget)
		self.photo_button.setObjectName("photo_button")
		self.photo_video_layout.addWidget(self.photo_button)
		self.video_button = QtWidgets.QToolButton(self.horizontalLayoutWidget)
		self.video_button.setObjectName("video_button")
		self.photo_video_layout.addWidget(self.video_button)


		self.verticalLayoutWidget_3 = QtWidgets.QWidget(Form)
		self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(0, 180, 141, 161))
		self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
		self.blurry_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
		self.blurry_layout.setContentsMargins(0, 0, 0, 0)
		self.blurry_layout.setObjectName("blurry_layout")
		self.blurry_option = QtWidgets.QToolButton(self.verticalLayoutWidget_3)
		self.blurry_option.setObjectName("blurry_option")
		self.blurry_layout.addWidget(self.blurry_option, 0, QtCore.Qt.AlignHCenter)
		self.blurry_people_button = QtWidgets.QRadioButton(self.verticalLayoutWidget_3)
		self.blurry_people_button.setObjectName("blurry_people_button")
		self.blurry_layout.addWidget(self.blurry_people_button, 0, QtCore.Qt.AlignHCenter)
		self.blurry_boxes_button = QtWidgets.QRadioButton(self.verticalLayoutWidget_3)
		self.blurry_boxes_button.setObjectName("blurry_boxes_button")
		self.blurry_layout.addWidget(self.blurry_boxes_button, 0, QtCore.Qt.AlignHCenter)


		self.verticalLayoutWidget_4 = QtWidgets.QWidget(Form)
		self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(0, 340, 141, 171))
		self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
		self.conversion_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
		self.conversion_layout.setContentsMargins(0, 0, 0, 0)
		self.conversion_layout.setObjectName("conversion_layout")
		self.detection_option = QtWidgets.QToolButton(self.verticalLayoutWidget_4)
		self.detection_option.setObjectName("detection_option")
		self.conversion_layout.addWidget(self.detection_option, 0, QtCore.Qt.AlignHCenter)
		self.face_key_button = QtWidgets.QRadioButton(self.verticalLayoutWidget_4)
		self.face_key_button.setObjectName("face_key_button")
		self.conversion_layout.addWidget(self.face_key_button, 0, QtCore.Qt.AlignHCenter)
		self.biggest_button = QtWidgets.QRadioButton(self.verticalLayoutWidget_4)
		self.biggest_button.setObjectName("biggest_button")
		self.conversion_layout.addWidget(self.biggest_button, 0, QtCore.Qt.AlignHCenter)

		self.verticalLayoutWidget_5 = QtWidgets.QWidget(Form)
		self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(0, 650, 141, 101))
		self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
		self.blurry_layout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
		self.blurry_layout_2.setContentsMargins(0, 0, 0, 0)
		self.blurry_layout_2.setObjectName("blurry_layout_2")
		self.reset = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
		self.reset.setObjectName("reset")
		self.exit = QtWidgets.QToolButton(self.verticalLayoutWidget_5)
		self.exit.setObjectName("exit")
		self.blurry_layout_2.addWidget(self.reset, 0, QtCore.Qt.AlignHCenter)
		self.blurry_layout_2.addWidget(self.exit, 0, QtCore.Qt.AlignHCenter)

		self.exit.clicked.connect(self.exit_button_clicked)
		self.reset.clicked.connect(self.reset_button_clicked)

		self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
		self.label.setMaximumSize(QtCore.QSize(980, 600))
		self.label.setObjectName("label")

		self.img_path = ""
		self.img = None

		self.mode = "1"

		self.model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		self.model1.eval()
		self.model2 = models.segmentation.fcn_resnet101(pretrained=True)
		self.model2.eval()
		self.model3 = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
		self.model3.eval()

		self.COCO_INSTANCE_CATEGORY_NAMES = [
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

		self.buttons={}
		self.buttons[self.photo_button] = 0
		self.buttons[self.video_button] = 0
		self.buttons[self.blurry_people_button] = 0
		self.buttons[self.blurry_boxes_button] = 0
		self.buttons[self.face_key_button] = 0
		self.buttons[self.biggest_button] = 0

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.upload_button.clicked.connect(self.upload_button_clicked)
		for button in self.buttons.keys():
			button.clicked.connect(partial(self.button_clicked_colored, button, self.buttons[button]))
		self.retranslateUi(Form)
		QtCore.QMetaObject.connectSlotsByName(Form)

	def object_detection_api(self, img_path, mode, threshold=0.5, rect_th=2, text_size=1.5, text_th=2): 
		print('mode', mode)
		img = Image.open(img_path) # Load the image
		transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
		img = transform(img) # Apply the transform to the image
		pred = self.model3([img]) # Pass the image to the model
		pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]  
		pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
		pred_score = list(pred[0]['scores'].detach().numpy())
		pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
		pred_boxes = pred_boxes[:pred_t+1]
		pred_class = pred_class[:pred_t+1]
		masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
		masks = masks[:pred_t+1]
		boxes = pred_boxes
		pred_cls = pred_class

		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

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
				try:
					cv2.rectangle(img, box_coord_dict[box_idx][0], box_coord_dict[box_idx][1],color=(103, 142, 240), thickness= rect_th) # Draw Rectangle with the coordinates
				except:
					print('i=0 error')
			else:
				try:
					crop_img = img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])]
					noised_img = cv2.blur(crop_img,(7,7),0)
					img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])] = noised_img
				except:
					print('other i error')
		if mode == "1":
			height, width, channel = img.shape
			bytesPerLine = 3 * width
			qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
			self.pixmap = QPixmap(qImg)
			self.label.setPixmap(self.pixmap)
			self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
		
		elif mode == "2":
			blurred_img  = img
			blurred_bgr = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR) # rgb 조정
			original_img = cv2.imread(img_path) # original

			img = Image.open(img_path)
			trf = T.Compose([T.ToTensor()])
			inp = trf(img).unsqueeze(0).to(self.device)
			out = self.model2.to(self.device)(inp)['out']
			om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
			iteration = 0

			for i in range(om.shape[0]):
				for j in range(om.shape[1]):
					iteration +=1
					if om[i,j] == 15:
						original_img[i,j,:] = blurred_bgr[i,j,:] # 사람인 pixel에 blur pixel 복사
			height, width, channel = blurred_bgr.shape
			bytesPerLine = 3 * width
			qImg = QImage(blurred_bgr.data, width, height, bytesPerLine, QImage.Format_RGB888)
			self.pixmap = QPixmap(qImg)
			self.label.setPixmap(self.pixmap)
			self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)


		elif mode == "3":
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			for i in range(len(masks)):
				if pred_cls[i]=='person':
					colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
					r = np.zeros_like(masks[i]).astype(np.uint8);	g = np.zeros_like(masks[i]).astype(np.uint8);	b = np.zeros_like(masks[i]).astype(np.uint8)
					r[masks[i] == 1], g[masks[i] == 1], b[masks[i] == 1] = colours[random.randrange(0,10)]
					coloured_mask = np.stack([r, g, b], axis=2)
					rgb_mask = coloured_mask
					img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
					cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
					person_index = pred_cls[i]+' '+str(i)
					cv2.putText(img,person_index, boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

			height, width, channel = img.shape
			bytesPerLine = 3 * width
			qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
			self.pixmap = QPixmap(qImg)
			self.label.setPixmap(self.pixmap)
			self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)

		else:
			print('error')
		self.conversion_button.setParent(None)


	def upload_button_clicked(self):  
		_translate = QtCore.QCoreApplication.translate
		fname = QFileDialog.getOpenFileName(Form)
		self.image_path = fname[0]
		self.label.setParent(None)
		self.pixmap = QPixmap(self.image_path)
		self.label.setPixmap(self.pixmap)
		self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
		self.upload_button.setParent(None)

		self.conversion_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
		self.conversion_button.setMaximumSize(QtCore.QSize(300, 16777215))
		self.conversion_button.setObjectName("upload_button")
		self.conversion_button.setText(_translate("Form", "Start Conversion"))
		self.resultLayout.addWidget(self.conversion_button, 0, QtCore.Qt.AlignHCenter)
		self.conversion_button.clicked.connect(partial(self.object_detection_api, self.image_path, self.mode, threshold=0.8))

	def exit_button_clicked(self):
		sys.exit(QtWidgets.QApplication(sys.argv).exec_())

	def reset_button_clicked(self):
		_translate = QtCore.QCoreApplication.translate
		self.label.setParent(None)
		self.upload_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
		self.upload_button.setMaximumSize(QtCore.QSize(300, 16777215))
		self.upload_button.setObjectName("upload_button")
		self.upload_button.setText(_translate("Form", "Upload your file here!"))
		self.resultLayout.addWidget(self.upload_button, 0, QtCore.Qt.AlignHCenter)

		print('in reset function')
		for button in self.buttons.keys():
			print(button)
			if button == self.photo_button or button == self.video_button: 
				self.buttons[button] = 0
				button.setStyleSheet("background-color: light gray")
			else:
				print('here i am ')
				print(button)
				print('before', button.isChecked())
				button.setChecked(False)
				print('after', button.isChecked())

		# for button in self.buttons.keys():
		# 	if button == self.photo_button or button == self.video_button: 
		# 		self.buttons[button] = 0
		# 		button.setStyleSheet("background-color: light gray")
		# 	else:
		# 		print('here i am ')
		# 		print(button)
		# 		print('before', button.isChecked())
		# 		button.setChecked(False)
		# 		print('after', button.isChecked())

		# self.label.setParent(None)
		# self.upload_button.setParent(None)
		# self.upload_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
		# self.upload_button.setMaximumSize(QtCore.QSize(300, 16777215))
		# self.upload_button.setObjectName("upload_button")
		# self.upload_button.setText(_translate("Form", "Upload your file here!"))
		# self.resultLayout.addWidget(self.upload_button, 0, QtCore.Qt.AlignHCenter)


	def button_clicked_colored(self, button, value):
		if button == self.photo_button or button == self.video_button: 
			if self.buttons[button] % 2 ==  0:
				button.setStyleSheet("background-color: #c5d7e8")
			else:
				button.setStyleSheet("background-color: light gray")
		else:
			pass
		self.buttons[button] += 1 

	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("Form", "Object Detection for No Object"))
		self.upload_button.setText(_translate("Form", "Upload your file here!"))
		self.photo_button.setText(_translate("Form", "        Photo        "))
		self.video_button.setText(_translate("Form", "        Video         "))
		self.blurry_people_button.setText(_translate("Form", " People"))
		self.blurry_boxes_button.setText(_translate("Form", " Boxes"))
		self.face_key_button.setText(_translate("Form", "Face-Key Point"))
		self.biggest_button.setText(_translate("Form", "Biggest Person"))
		self.reset.setText(_translate("Form", "        Reset         "))
		self.exit.setText(_translate("Form", "        Exit         "))
		self.blurry_option.setText(_translate("Form", "    OPTION1     "))
		self.detection_option.setText(_translate("Form", "    OPTION2     "))


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	Form = QtWidgets.QWidget()
	ui = Ui_Form()
	ui.setupUi(Form)
	Form.show()
	sys.exit(app.exec_())
