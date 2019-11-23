# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BlurryConverter_rev1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1001, 753)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(0, 180, 141, 161))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.blurry_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.blurry_layout.setContentsMargins(0, 0, 0, 0)
        self.blurry_layout.setObjectName("blurry_layout")
        self.blurry_option = QtWidgets.QToolButton(self.verticalLayoutWidget_3)
        self.blurry_option.setObjectName("blurry_option")
        self.blurry_layout.addWidget(self.blurry_option, 0, QtCore.Qt.AlignHCenter)
        self.blurry_people_button_2 = QtWidgets.QRadioButton(self.verticalLayoutWidget_3)
        self.blurry_people_button_2.setObjectName("blurry_people_button_2")
        self.blurry_layout.addWidget(self.blurry_people_button_2, 0, QtCore.Qt.AlignHCenter)
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
        self.straight_button = QtWidgets.QRadioButton(self.verticalLayoutWidget_4)
        self.straight_button.setObjectName("straight_button")
        self.conversion_layout.addWidget(self.straight_button, 0, QtCore.Qt.AlignHCenter)
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
        self.blurry_layout_2.addWidget(self.reset)
        self.exit = QtWidgets.QToolButton(self.verticalLayoutWidget_5)
        self.exit.setObjectName("exit")
        self.blurry_layout_2.addWidget(self.exit, 0, QtCore.Qt.AlignHCenter)
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

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.blurry_option.setText(_translate("Form", "OPTION 1"))
        self.blurry_people_button_2.setText(_translate("Form", "People"))
        self.blurry_boxes_button.setText(_translate("Form", "Boxes"))
        self.detection_option.setText(_translate("Form", "OPTION 2"))
        self.straight_button.setText(_translate("Form", "Straight"))
        self.biggest_button.setText(_translate("Form", "Biggest"))
        self.reset.setText(_translate("Form", "Reset"))
        self.exit.setText(_translate("Form", "         Exit          "))
        self.photo_button.setText(_translate("Form", "PHOTO"))
        self.video_button.setText(_translate("Form", "VIDEO"))
        self.upload_button.setText(_translate("Form", "Upload your file here!"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
