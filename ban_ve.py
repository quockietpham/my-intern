# Form implementation generated from reading ui file 'ban_ve.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(250, 10, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 70, 751, 101))
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 30, 71, 16))
        self.label_2.setObjectName("label_2")
        self.name = QtWidgets.QLineEdit(parent=self.groupBox)
        self.name.setGeometry(QtCore.QRect(100, 30, 113, 22))
        self.name.setObjectName("name")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 70, 71, 16))
        self.label_3.setObjectName("label_3")
        self.gioi_tinh = QtWidgets.QComboBox(parent=self.groupBox)
        self.gioi_tinh.setGeometry(QtCore.QRect(100, 70, 73, 22))
        self.gioi_tinh.setObjectName("gioi_tinh")
        self.gioi_tinh.addItem("")
        self.gioi_tinh.addItem("")
        self.gioi_tinh.addItem("")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 130, 751, 301))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(9, 199, 751, 341))
        self.groupBox_3.setObjectName("groupBox_3")
        self.CAFE = QtWidgets.QCheckBox(parent=self.groupBox_3)
        self.CAFE.setGeometry(QtCore.QRect(50, 40, 81, 20))
        self.CAFE.setObjectName("CAFE")
        self.BIMBIM = QtWidgets.QCheckBox(parent=self.groupBox_3)
        self.BIMBIM.setGeometry(QtCore.QRect(340, 40, 81, 20))
        self.BIMBIM.setObjectName("BIMBIM")
        self.NGUOILON = QtWidgets.QSpinBox(parent=self.groupBox_3)
        self.NGUOILON.setGeometry(QtCore.QRect(130, 130, 42, 22))
        self.NGUOILON.setObjectName("NGUOILON")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(40, 130, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(40, 190, 55, 16))
        self.label_5.setObjectName("label_5")
        self.TREEM = QtWidgets.QSpinBox(parent=self.groupBox_3)
        self.TREEM.setGeometry(QtCore.QRect(120, 190, 42, 22))
        self.TREEM.setObjectName("TREEM")
        self.TINHTIEN = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.TINHTIEN.setGeometry(QtCore.QRect(260, 240, 93, 28))
        self.TINHTIEN.setObjectName("TINHTIEN")
        self.lineEdit = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(380, 250, 113, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.VND = QtWidgets.QLabel(parent=self.groupBox_3)
        self.VND.setGeometry(QtCore.QRect(510, 250, 55, 16))
        self.VND.setObjectName("VND")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.TINHTIEN.clicked.connect(self.tt)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "ỨNG DỤNG BÁN VÉ"))
        self.groupBox.setTitle(_translate("MainWindow", "THÔNG TIN KHÁCH HÀNG"))
        self.label_2.setText(_translate("MainWindow", "HỌ VÀ TÊN:"))
        self.label_3.setText(_translate("MainWindow", "GIỚI TÍNH:"))
        self.gioi_tinh.setItemText(0, _translate("MainWindow", "nam"))
        self.gioi_tinh.setItemText(1, _translate("MainWindow", "nu"))
        self.gioi_tinh.setItemText(2, _translate("MainWindow", "bd"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.groupBox_3.setTitle(_translate("MainWindow", "DICH VU"))
        self.CAFE.setText(_translate("MainWindow", "CAFE"))
        self.BIMBIM.setText(_translate("MainWindow", "BIMBIM"))
        self.label_4.setText(_translate("MainWindow", "NGUOI LON"))
        self.label_5.setText(_translate("MainWindow", "TRE EM:"))
        self.TINHTIEN.setText(_translate("MainWindow", "TINH TIEN"))
        self.VND.setText(_translate("MainWindow", "VND"))
    def tt(self):
        price={
            "CAFE":100000,
            "BIMBIM":50000,
            "NGUOILON":200000,
            "TREEM":50000
        }
        tcf=0
        if self.CAFE.checkState()!=0:
            tcf=price(['CAFE'])
        tbb=0



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
