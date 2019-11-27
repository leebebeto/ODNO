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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(999, 751)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(140, 0, 861, 751))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.resultLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.resultLayout.setContentsMargins(0, 0, 0, 0)
        self.resultLayout.setObjectName("resultLayout")

        self.upload_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.upload_button.setMaximumSize(QtCore.QSize(300, 16777215))
        self.upload_button.setObjectName("upload_button")
        self.resultLayout.addWidget(self.upload_button, 0, QtCore.Qt.AlignHCenter)

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 141, 121))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.photo_video_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.photo_video_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_video_layout.setObjectName("photo_video_layout")
        self.photo_button = QtWidgets.QToolButton(self.verticalLayoutWidget_2)
        self.photo_button.setObjectName("photo_button")
        self.photo_video_layout.addWidget(self.photo_button, 0, QtCore.Qt.AlignHCenter)
        self.video_button = QtWidgets.QToolButton(self.verticalLayoutWidget_2)
        self.video_button.setObjectName("video_button")
        self.photo_video_layout.addWidget(self.video_button, 0, QtCore.Qt.AlignHCenter)

        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(0, 120, 141, 161))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.blurry_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.blurry_layout.setContentsMargins(0, 0, 0, 0)
        self.blurry_layout.setObjectName("blurry_layout")
        self.blurry_people_button = QtWidgets.QToolButton(self.verticalLayoutWidget_3)
        self.blurry_people_button.setObjectName("blurry_people_button")
        self.blurry_layout.addWidget(self.blurry_people_button, 0, QtCore.Qt.AlignHCenter)
        self.blurry_boxes_button = QtWidgets.QToolButton(self.verticalLayoutWidget_3)
        self.blurry_boxes_button.setObjectName("blurry_boxes_button")
        self.blurry_layout.addWidget(self.blurry_boxes_button, 0, QtCore.Qt.AlignHCenter)

        self.verticalLayoutWidget_4 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(0, 280, 141, 171))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.conversion_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.conversion_layout.setContentsMargins(0, 0, 0, 0)
        self.conversion_layout.setObjectName("conversion_layout")
        self.face_key_point_button = QtWidgets.QToolButton(self.verticalLayoutWidget_4)
        self.face_key_point_button.setObjectName("face_key_point_button")
        self.conversion_layout.addWidget(self.face_key_point_button, 0, QtCore.Qt.AlignHCenter)
        self.biggest_button = QtWidgets.QToolButton(self.verticalLayoutWidget_4)
        self.biggest_button.setObjectName("biggest_button")
        self.conversion_layout.addWidget(self.biggest_button, 0, QtCore.Qt.AlignHCenter)

        self.verticalLayoutWidget_5 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(0, 680, 141, 71))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.blurry_layout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.blurry_layout_2.setContentsMargins(0, 0, 0, 0)
        self.blurry_layout_2.setObjectName("blurry_layout_2")
        self.exit = QtWidgets.QToolButton(self.verticalLayoutWidget_5)
        self.exit.setObjectName("exit")
        self.reset = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.reset.setObjectName("reset")
        self.blurry_layout_2.addWidget(self.reset)
        self.blurry_layout_2.addWidget(self.exit, 0, QtCore.Qt.AlignHCenter)

        self.exit.clicked.connect(self.exit_button_clicked)
        self.reset.clicked.connect(self.reset_button_clicked)

        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setMaximumSize(QtCore.QSize(980, 600))
        self.label.setObjectName("label")

        self.img_path = ""
        self.img = None

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
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
        self.buttons[self.face_key_point_button] = 0
        self.buttons[self.biggest_button] = 0

        self.upload_button.clicked.connect(self.upload_button_clicked)
        for button in self.buttons.keys():
            button.clicked.connect(partial(self.button_clicked_colored, button, self.buttons[button]))
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def test(self):
        print("hello world")

    def sample_img(self, img_path):
        list_img_data = os.listdir(img_data)  
        list_img_data.sort()
        img_ex_path = img_data + list_img_data[98]
        img_ex_origin = cv2.imread(img_ex_path)
        return img_ex_path

    def object_detection_api(self, img_path, threshold=0.5, rect_th=3, text_size=1.5, text_th=3): 
        img = Image.open(img_path) # Load the image
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        pred = self.model([img]) # Pass the image to the model
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]  
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] 
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        boxes = pred_boxes
        pred_cls = pred_class
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        for i in range(len(boxes)):
            left_x = round(boxes[i][0][0])
            left_y = round(boxes[i][0][1])
            right_x = round(boxes[i][1][0])
            right_y = round(boxes[i][1][1])
            # print(left_x, left_y, right_x, right_y)
            try:
                pass
                # cv2.rectangle(img, (int(left_x), int(left_y)), (int(right_x), int(right_y)), color=(103, 142, 240), thickness=rect_th) # Draw Rectangle with the coordinates
            except:
                pass
            # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
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
                    pass
                    # cv2.rectangle(img, box_coord_dict[box_idx][0], box_coord_dict[box_idx][1],color=(103, 142, 240), thickness= rect_th) # Draw Rectangle with the coordinates
                except:
                    print('i=0 error')
            else:
                try:
                    crop_img = img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])]
                    noised_img = cv2.blur(crop_img,(7,7),0)
                    img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])] = noised_img
                    # cv2.rectangle(img, box_coord_dict[box_size_dict[i][0]][0], box_coord_dict[box_size_dict[i][0]][1], color=(103, 142, 240), thickness=rect_th) # Draw Rectangle with the coordinates
                except:
                    print('other i error')

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap(qImg)
        self.label.setPixmap(self.pixmap)
        self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)

        # for i in range(len(boxes)):
        #     # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        #     # cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
        # box_size_dict = {}
        # box_coord_dict = {}
        # for i in range(len(boxes)):
        #     x_len = abs(boxes[i][0][0] - boxes[i][1][0])
        #     y_len = abs(boxes[i][0][1] - boxes[i][1][1])
        #     box_size = x_len * y_len
        #     box_size_dict[i] = box_size
        #     box_coord_dict[i] = boxes[i] 
        # box_size_dict = sorted(box_size_dict.items(), key = lambda x: x[1], reverse = True)
        # for i in range(len(box_size_dict)):
        #     box_idx = box_size_dict[i][0]
        #     if i ==0:       
        #         cv2.rectangle(img, box_coord_dict[box_idx][0], box_coord_dict[box_idx][1],color=(0, 255, 0), thickness= 3) # Draw Rectangle with the coordinates
        #     else:
        #         crop_img = img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])]
        #         noised_img = cv2.blur(crop_img,(7,7),0)
        #         img[int(boxes[box_idx][0][1]):int(boxes[box_idx][1][1]), int(boxes[box_idx][0][0]):int(boxes[box_idx][1][0])] = noised_img
        #         cv2.rectangle(img, box_coord_dict[box_size_dict[i][0]][0], box_coord_dict[box_size_dict[i][0]][1],color=(0, 255, 0), thickness= 3) # Draw Rectangle with the coordinates

        # height, width, channel = img.shape
        # bytesPerLine = 3 * width
        # qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        # self.pixmap = QPixmap(qImg)
        # self.label.setPixmap(self.pixmap)
        # self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)


     
    def decode_segmap(self, image, nc=21):
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

    def segment(self, net, device, path, show_orig=True):
      img = Image.open(path)
    #   trf = T.Compose([T.Resize(640), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
      trf = T.Compose([T.ToTensor()])
      inp = trf(img).unsqueeze(0).to(device)
      inp2 = trf(img).squeeze().to(device).detach().cpu().numpy()
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
        self.conversion_button.clicked.connect(partial(self.conversion, self.image_path))

    def exit_button_clicked(self):
        sys.exit(QtWidgets.QApplication(sys.argv).exec_())

    def reset_button_clicked(self):
        for button in self.buttons.keys():
            self.buttons[button] = 0
            button.setStyleSheet("background-color: light gray")
        self.test()

    def button_clicked_colored(self, button, value):
        if self.buttons[button] % 2 ==  0:
            button.setStyleSheet("background-color: #c5d7e8")
        else:
            button.setStyleSheet("background-color: light gray")
        self.buttons[button] += 1 


    def conversion(self, img_path):
        print('before conversion')
        self.conversion_button.clicked.connect(partial(self.object_detection_api, img_path, threshold=0.8))
        # self.label.setParent(None)
        # self.upload_button.setParent(None)
        # img, boxes, pred_cls = partial(self.object_detection_api, img_ex_path, threshold=0.8)
        # img_ex_path = fname[0]
        # print(img_ex_path)
        # print(img)
        # redundant_blurry(img, boxes, pred_cls)



    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Object Detection for No Object"))
        self.upload_button.setText(_translate("Form", "Upload your file here!"))
        self.photo_button.setText(_translate("Form", "        Photo        "))
        self.video_button.setText(_translate("Form", "        Video         "))
        self.blurry_people_button.setText(_translate("Form", " Blurry with people"))
        self.blurry_boxes_button.setText(_translate("Form", " Blurry with boxes"))
        self.face_key_point_button.setText(_translate("Form", "  Face Key Point  "))
        self.biggest_button.setText(_translate("Form", "       Biggest       "))
        self.reset.setText(_translate("Form", "Reset"))
        self.exit.setText(_translate("Form", "         Exit          "))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
