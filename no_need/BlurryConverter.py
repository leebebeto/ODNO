from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import *

class Ui_Form(object):
    def conversion(self):
        self.label.setParent(None)
        self.pixmap = QPixmap('after.png')
        self.label.setPixmap(self.pixmap)
        self.verticalLayout.addWidget(self.label)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(675, 435)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 671, 101))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.toolButton_6 = QtWidgets.QToolButton(self.horizontalLayoutWidget)
        self.toolButton_6.setObjectName("toolButton_6")
        self.horizontalLayout_2.addWidget(self.toolButton_6)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 100, 671, 331))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QLabel()
        self.pixmap = QPixmap('before.png')
        self.label.setPixmap(self.pixmap)
        self.verticalLayout.addWidget(self.label)
        # self.resize(pixmap.width(),pixmap.height())
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.toolButton_6.setText(_translate("Form", "Start Conversion!"))
        self.toolButton_6.clicked.connect(self.conversion)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
