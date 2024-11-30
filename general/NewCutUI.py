import sys
import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QAction, QFileDialog

from GraphMaker import GraphMaker


class NewCutUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graph_maker = GraphMaker()
        self.setWindowTitle("GraphCut")
        self.seed_num = self.graph_maker.foreground

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')

        openButton = QAction(QIcon('exit24.png'), 'Open Image', self)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open a file for segmenting.')
        openButton.triggered.connect(self.on_open)
        fileMenu.addAction(openButton)

        saveButton = QAction(QIcon('exit24.png'), 'Save Image', self)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save file to disk.')
        saveButton.triggered.connect(self.on_save)
        fileMenu.addAction(saveButton)

        closeButton = QAction(QIcon('exit24.png'), 'Exit', self)
        closeButton.setShortcut('Ctrl+Q')
        closeButton.setStatusTip('Exit application')
        closeButton.triggered.connect(self.on_close)
        fileMenu.addAction(closeButton)

        self.setMenuBar(mainMenu)

        # Setup main widget
        mainWidget = QWidget()
        mainBox = QVBoxLayout()

        # Setup Mode Buttons
        buttonLayout = QHBoxLayout()
        self.foregroundButton = QPushButton('Add Foreground Seeds')
        self.foregroundButton.clicked.connect(self.on_foreground)
        self.foregroundButton.setStyleSheet("background-color: gray")

        self.backGroundButton = QPushButton('Add Background Seeds')
        self.backGroundButton.clicked.connect(self.on_background)
        self.backGroundButton.setStyleSheet("background-color: white")

        clearButton = QPushButton('Clear All Seeds')
        clearButton.clicked.connect(self.on_clear)

        segmentButton = QPushButton('Segment Image')
        segmentButton.clicked.connect(self.on_segment)

        buttonLayout.addWidget(self.foregroundButton)
        buttonLayout.addWidget(self.backGroundButton)
        buttonLayout.addWidget(clearButton)
        buttonLayout.addWidget(segmentButton)
        buttonLayout.addStretch()

        mainBox.addLayout(buttonLayout)

        # Setup Image Area
        imageLayout = QHBoxLayout()

        self.seedLabel = QLabel()
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag

        self.segmentLabel = QLabel()

        imageLayout.addWidget(self.seedLabel)
        imageLayout.addWidget(self.segmentLabel)
        imageLayout.addStretch()
        mainBox.addLayout(imageLayout)

        mainBox.addStretch()
        mainWidget.setLayout(mainBox)
        self.setCentralWidget(mainWidget)

    def run(self):
        self.show()

    @staticmethod
    def get_qimage(cvimage):
        if cvimage is None:
            return QImage()

        cvimage_rgb = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
        height, width, bytes_per_pix = cvimage_rgb.shape
        bytes_per_line = width * bytes_per_pix
        return QImage(cvimage_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)


    @pyqtSlot()
    def on_foreground(self):
        self.seed_num = self.graph_maker.foreground
        self.foregroundButton.setStyleSheet("background-color: gray")
        self.backGroundButton.setStyleSheet("background-color: white")

    @pyqtSlot()
    def on_background(self):
        self.seed_num = self.graph_maker.background
        self.foregroundButton.setStyleSheet("background-color: white")
        self.backGroundButton.setStyleSheet("background-color: gray")

    @pyqtSlot()
    def on_clear(self):
        self.graph_maker.clear_seeds()
        self.update_image()

    @pyqtSlot()
    def on_segment(self):
        self.graph_maker.create_graph()
        self.update_image()

    @pyqtSlot()
    def on_open(self):
        print("Open button clicked")
        f, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)")
        print(f)  # Debug print to check file path
        if f:
            self.graph_maker.load_image(f)
            if self.graph_maker.image is not None:
                self.seedLabel.setPixmap(QPixmap.fromImage(
                    self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
                self.segmentLabel.setPixmap(QPixmap.fromImage(
                    self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.segmented))))
            else:
                print("Failed to load image")
        else:
            print("No file selected")


    @pyqtSlot()
    def on_save(self):
        f, _ = QFileDialog.getSaveFileName()
        if f:
            self.graph_maker.save_image(f)

    @pyqtSlot()
    def on_close(self):
        self.close()

    def mouse_down(self, event):
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_num)
        self.update_image()

    def mouse_drag(self, event):
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_num)
        self.update_image()

    def update_image(self):
        # Update the seed and segment images in the labels
        seed_image = self.graph_maker.get_image_with_overlay(self.graph_maker.seeds)
        segmented_image = self.graph_maker.get_image_with_overlay(self.graph_maker.segmented)
        
        self.seedLabel.setPixmap(QPixmap.fromImage(self.get_qimage(seed_image)))
        self.segmentLabel.setPixmap(QPixmap.fromImage(self.get_qimage(segmented_image)))
        
        # Redraw the labels
        self.seedLabel.update()
        self.segmentLabel.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    newUI = NewCutUI()
    newUI.run()
    sys.exit(app.exec_())
