import os, sys, glob, matplotlib
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.backends.qt_compat import *
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import test

root=str(os.path.dirname(os.path.abspath(__file__)))
dirText=''
dir=root + '\\data\\unseen\\demo1.jpg'

def predict():
    res, _, img = test.predictSingleUnseen(dir)
    print(res)
    pred.setText(res)
    pred.show()
    preview.setPixmap(QPixmap(dir))
    preview.show()

def applyDir():
    global dir
    [_dir, _] = QFileDialog.getOpenFileName(None, 'Select image:', '.', "Image files (*.jpg)")
    dir=_dir
    dirText.setText('Current dir: ' + dir)
    dirText.show()
    preview.setPixmap(QPixmap(dir))
    preview.show()
    pred.setText('')
    pred.show()

def initUI():
    global dirText, preview, pred
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Trash Classification - Prethesis")
    window.setGeometry(QRect(0,0, 960, 720))

    preview=QLabel()
    preview.setPixmap(QPixmap(root + '\\data\\unseen\\demo1.jpg'))
    preview.setScaledContents(True)
    preview.setStyleSheet("QLabel { background-color: black; border: 2px dashed red; height: 450px; aspect-ratio: 3 / 2; max-width: 600px; max-height: 600px;}")

    dirText=QLabel()
    dirText.setText('Current dir: '+ root + '\\data\\unseen\\demo1.jpg')
    dirText.setStyleSheet("QLabel { color: black; font-size: 28px; padding: 10px;  border-radius: 8px;}")
    dirText.setScaledContents(True)

    pred=QLabel("Empty")
    pred.setStyleSheet("QLabel { color: black; font-size: 36px; padding: 10px; border-radius: 8px;}")
    pred.setScaledContents(True)
    predict()

    dirButton=QPushButton("Choose image")
    dirButton.setStyleSheet("QPushButton { background: ##F3F3ED; font-size: 36px; padding: 10px 0; border-radius: 8px; text-align: center; text-decoration: underline; width: fit-content; height: 60px;}")
    dirButton.clicked.connect(applyDir)

    predictButton=QPushButton("Predict")
    predictButton.setStyleSheet("QPushButton { background: ##F3F3ED; font-size: 36px; padding: 10px 0; border-radius: 8px; text-align: center; width: 60px; height: 60px;}")
    predictButton.clicked.connect(predict)

    predictClass=QLabel("Predict Class")
    predictClass.setStyleSheet("QLabel { font-size: 36px; padding: 10px; border-radius: 8px; text-decoration: underline; width: 100%; text-transform: capitalize;}")
    predictClass.setScaledContents(True)

    layout = QGridLayout()
    layout.addWidget(preview, 0, 0, 2, 1)
    layout.addWidget(predictClass, 0, 1)
    layout.addWidget(pred, 1, 1)
    layout.addWidget(predictButton, 2, 0)
    layout.addWidget(dirButton, 2, 1)
    layout.addWidget(dirText, 3, 0, 1, 2)
    layout.addWidget(QLabel('Prethesis: Trash classification by Pham Cong Tuan ITITIU19060'), 4,0,1,2)

    window.setLayout(layout)
    window.show()
    sys.exit(app.exec())

initUI()