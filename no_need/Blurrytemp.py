# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BlurryConverter_rev.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(997, 753)
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
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setMaximumSize(QtCore.QSize(600, 500))
        self.label.setText("")
        self.label.setObjectName("label")
        self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
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
        self.conversion_layout.addWidget(self.face_key_point_button)
        self.biggest_button = QtWidgets.QToolButton(self.verticalLayoutWidget_4)
        self.biggest_button.setObjectName("biggest_button")
        self.conversion_layout.addWidget(self.biggest_button)
        self.conversion_button = QtWidgets.QToolButton(self.verticalLayoutWidget_4)
        self.conversion_button.setObjectName("conversion_button")
        self.conversion_layout.addWidget(self.conversion_button)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(0, 650, 141, 101))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.blurry_layout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.blurry_layout_2.setContentsMargins(0, 0, 0, 0)
        self.blurry_layout_2.setObjectName("blurry_layout_2")
        self.reset = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.reset.setObjectName("reset")
        self.blurry_layout_2.addWidget(self.reset)
        self.exit = QtWidgets.QToolButton(self.verticalLayoutWidget_5)
        self.exit.setObjectName("exit")
        self.blurry_layout_2.addWidget(self.exit, 0, QtCore.Qt.AlignHCenter)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.upload_button.setText(_translate("Form", "Upload your file here!"))
        self.photo_button.setText(_translate("Form", "        Photo        "))
        self.video_button.setText(_translate("Form", "        Video         "))
        self.blurry_people_button.setText(_translate("Form", " Blurry with people"))
        self.blurry_boxes_button.setText(_translate("Form", " Blurry with boxes"))
        self.face_key_point_button.setText(_translate("Form", "  Face Key Point  "))
        self.biggest_button.setText(_translate("Form", "       Biggest       "))
        self.conversion_button.setText(_translate("Form", " Start Conversion!"))
        self.reset.setText(_translate("Form", "Reset"))
        self.exit.setText(_translate("Form", "         Exit          "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


#   # for i in range(len(boxes)):
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

