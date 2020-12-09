import sys
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QListView
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QHBoxLayout


class Ui_ImageWindow(object):
    def setupUi(self, ImageWindow):
        ImageWindow.setObjectName("MainWindow")
        ImageWindow.resize(1220, 820)
        self.centralwidget = QtWidgets.QWidget()

        self.centralwidget.setObjectName("centralwidget")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(20, 20, 1180, 780))
        self.listWidget.setFlow(QListView.TopToBottom)
        self.listWidget.setObjectName("listWidget")

        ImageWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(ImageWindow)
        QtCore.QMetaObject.connectSlotsByName(ImageWindow)
        self.setWindowTitle('图片列表')
        # self.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图片列表"))


class ImageWindow(QMainWindow, Ui_ImageWindow):
    def __init__(self, dirts, imgName):
        self.dirts = dirts
        self.imgName = imgName
        self.imageHeight = 250
        super(ImageWindow, self).__init__()
        self.setupUi(self)
        self.addList()

    def open(self):
        self.show()

    def addList(self):
        for dirt in self.dirts:
            files = os.listdir(dirt)
            images = []
            print(files)
            for file in files:
                if file.startswith(self.imgName):
                    images.append(file)
            self.addImg(dirt, images)

    def imgWidget(self, dirt, image):
        # 总Widget
        wight = QWidget()
        mianLayout = QVBoxLayout()
        file = dirt + '/' + image
        pixmap = QPixmap(file)
        width = 0
        if not pixmap.isNull():
            autoWidth = int(pixmap.width() * self.imageHeight / pixmap.height())
            width = autoWidth
            label = QtWidgets.QLabel(pixmap=pixmap)
            label.setScaledContents(True)
            label.setFixedHeight(self.imageHeight)
            label.setFixedWidth(autoWidth)
            mianLayout.addWidget(label, 0, Qt.AlignHCenter)
            mianLayout.addWidget(QLabel(image), 0, Qt.AlignHCenter)
        wight.setLayout(mianLayout)  # 布局给wight
        return wight, width  # 返回wight

    def addImg(self, dirt, images):
        rowWight = QWidget()
        rowListLayout = QVBoxLayout()
        rowListWidget = QListWidget()
        rowListWidget.resize(1100, 350)
        rowListWidget.setFlow(QListView.LeftToRight)
        # rowListWidget.setStyle(QStyle.Item)

        for image in images:
            widget, width = self.imgWidget(dirt, image)  # 调用上面的函数获取对应

            item = QListWidgetItem()  # 创建QListWidgetItem对象
            item.setSizeHint(QSize(width + 30, 300))  # 设置QListWidgetItem大小

            rowListWidget.addItem(item)  # 添加item
            rowListWidget.setItemWidget(item, widget)  # 为item设置widget

        rowListLayout.addWidget(QLabel(dirt))
        rowListLayout.addWidget(rowListWidget)
        rowWight.setLayout(rowListLayout)

        rowItem = QListWidgetItem()  # 创建QListWidgetItem对象
        rowItem.setSizeHint(QSize(400, 390))  # 设置QListWidgetItem大小
        self.listWidget.addItem(rowItem)  # 添加item
        self.listWidget.setItemWidget(rowItem, rowWight)  # 为item设置widget


class UI_MainWindow(object):

    def setupUI(self, MainWindow):
        MainWindow.setObjectName("ImageViewer")
        MainWindow.resize(550, 525)

        self.mainWidget = QtWidgets.QWidget(MainWindow)
        self.mainWidget.setObjectName("mainWidget")

        self.listWidget = QListWidget(self.mainWidget)
        self.listWidget.setGeometry(QtCore.QRect(10, 10, 370, 370))
        self.listWidget.setObjectName("ListWidget")

        self.addButton = QtWidgets.QPushButton(self.mainWidget)
        self.addButton.setGeometry(QtCore.QRect(400, 100, 120, 50))
        self.addButton.setObjectName("AddButton")

        self.clearButton = QtWidgets.QPushButton(self.mainWidget)
        self.clearButton.setGeometry(QtCore.QRect(400, 200, 120, 50))
        self.clearButton.setObjectName("ClearButton")

        self.fileNameEdit = QtWidgets.QLineEdit(self.mainWidget)
        self.fileNameEdit.setAlignment(Qt.AlignCenter)
        self.fileNameEdit.setFont(QFont('Arial', 18))
        self.fileNameEdit.setGeometry(QtCore.QRect(50, 400, 450, 45))
        self.fileNameEdit.setObjectName("FileNameEdit")

        self.searchButton = QtWidgets.QPushButton(self.mainWidget)
        self.searchButton.setGeometry(QtCore.QRect(100, 465, 120, 40))
        self.searchButton.setObjectName("SearchButton")

        self.resetButton = QtWidgets.QPushButton(self.mainWidget)
        self.resetButton.setGeometry(QtCore.QRect(330, 465, 120, 40))
        self.resetButton.setObjectName("ResetButton")

        MainWindow.setCentralWidget(self.mainWidget)
        self.translateUI(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def translateUI(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("ImageViewer", "图片查看"))
        self.addButton.setText(_translate("ImageViewer", "添加路径"))
        self.clearButton.setText(_translate("ImageViewer", "清空路径"))
        self.searchButton.setText(_translate("ImageViewer", "打开图片"))
        self.resetButton.setText(_translate("ImageViewer", "重置名称"))


class MainWindow(QMainWindow, UI_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUI(self)
        self.addButton.clicked.connect(self.addDir)
        self.clearButton.clicked.connect(self.clearDir)
        self.searchButton.clicked.connect(self.openImages)
        self.resetButton.clicked.connect(self.clearImageName)
        self.fileNameEdit.textChanged.connect(self.updateFileName)

        self.dirs = []
        self.imgName = ""

    def getItemWidget(self, dir, count):
        # 总Widget
        wight = QWidget()
        new_btn = QPushButton("删除")
        new_btn.setFixedSize(70, 30)

        # 总体横向布局
        layout_main = QHBoxLayout()

        # new_btn.clicked.connect(lambda: self.btn_college(self.sender().text()))

        # 按照从左到右, 从上到下布局添加
        layout_main.addWidget(QLabel(dir))  # 最左边的头像
        layout_main.addWidget(new_btn, 0, Qt.AlignVCenter)  # 最左边的头像

        wight.setLayout(layout_main)  # 布局给wight
        return wight  # 返回wight

    def refreshList(self):
        self.listWidget.clear()
        count = 0
        for dir in self.dirs:
            item = QListWidgetItem()  # 创建QListWidgetItem对象
            item.setSizeHint(QSize(370, 60))  # 设置QListWidgetItem大小
            widget = self.getItemWidget(dir, count)  # 调用上面的函数获取对应
            count += 1
            self.listWidget.addItem(item)  # 添加item
            self.listWidget.setItemWidget(item, widget)  # 为item设置widget

    def addDir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "F:\\")
        print(dir_path)
        if dir_path != '':
            self.dirs.append(dir_path)
            self.refreshList()

    def clearDir(self):
        self.dirs.clear()
        self.refreshList()

    def openImages(self):
        if len(self.dirs) > 0 and self.imgName != '':
            self.child_window = ImageWindow(self.dirs, self.imgName)
            self.child_window.show()

    def clearImageName(self):
        self.imgName = ''
        self.fileNameEdit.clear()

    def updateFileName(self, text):
        print('文件名称为：', text)
        self.imgName = text


def openImg(mainWindow):
    imageWindow = ImageWindow(mainWindow.dirs, mainWindow.imgName)
    return imageWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
