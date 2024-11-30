import sys
import logging
from typing import Optional, Tuple

import numpy as np
import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QIcon, QMouseEvent
from PyQt5.QtWidgets import (
    # QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QAction,
    QFileDialog,
    QMessageBox,
)

from Graph import Graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("graphcut.log")],
)
logger = logging.getLogger(__name__)


class GUI(QMainWindow):
    """
    A PyQt5-based user interface for graph-based image segmentation.

    Manages user interactions for adding seeds and segmenting images.
    """

    def __init__(self, graph: Optional[Graph] = None):
        """
        Initialize the UI.

        Args:
            graph (Optional[Graph]): An optional Graph instance
        """
        super().__init__()

        # Use provided graph or create a new instance
        self.graph = graph or Graph()

        # Current seed mode (foreground or background)
        self.seed_mode = self.graph.foreground

        # UI components
        self.seed_label: Optional[QLabel] = None
        self.segment_label: Optional[QLabel] = None
        self.foreground_button: Optional[QPushButton] = None
        self.background_button: Optional[QPushButton] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the entire user interface."""
        self.setWindowTitle("GraphCut Image Segmentation")
        self.resize(1200, 800)  # Give more space for image display

        # Create menu bar and actions
        self._create_menu_bar()

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create buttons and image areas
        button_layout = self._create_button_layout()
        image_layout = self._create_image_layout()

        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _create_menu_bar(self):
        """Create menu bar with file operations."""
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")

        actions = [
            ("Open", "Ctrl+O", self.on_open, "Open an image for segmentation"),
            ("Save", "Ctrl+S", self.on_save, "Save segmented image"),
            ("Exit", "Ctrl+Q", self.close, "Close application"),
        ]

        for title, shortcut, slot, status_tip in actions:
            action = QAction(QIcon("exit24.png"), title, self)
            action.setShortcut(shortcut)
            action.setStatusTip(status_tip)
            action.triggered.connect(slot)
            file_menu.addAction(action)

    def _create_button_layout(self) -> QHBoxLayout:
        """
        Create button layout for seed and segmentation controls.

        Returns:
            QHBoxLayout: Layout containing control buttons
        """
        button_layout = QHBoxLayout()

        # Foreground button
        self.foreground_button = QPushButton("Add Foreground Seeds")
        self.foreground_button.clicked.connect(self.on_foreground)
        self.foreground_button.setStyleSheet("background-color: white")

        # Background button
        self.background_button = QPushButton("Add Background Seeds")
        self.background_button.clicked.connect(self.on_background)
        self.background_button.setStyleSheet("background-color: white")

        # Other buttons
        buttons = [
            (self.foreground_button, "Add Foreground Seeds"),
            (self.background_button, "Add Background Seeds"),
            (QPushButton("Clear All Seeds"), "Clear All Seeds", self.on_clear),
            (QPushButton("Segment Image"), "Segment Image", self.on_segment),
        ]

        for button_config in buttons:
            if len(button_config) == 2:
                button, _ = button_config
                button_layout.addWidget(button)
            else:
                button, _, slot = button_config
                button.clicked.connect(slot)
                button_layout.addWidget(button)

        button_layout.addStretch()
        return button_layout

    def _create_image_layout(self) -> QHBoxLayout:
        """
        Create image display layout.

        Returns:
            QHBoxLayout: Layout containing seed and segmented image labels
        """
        image_layout = QHBoxLayout()

        self.seed_label = QLabel("Seed Image")
        self.seed_label.mousePressEvent = self.mouse_down
        self.seed_label.mouseMoveEvent = self.mouse_drag

        self.segment_label = QLabel("Segmented Image")

        image_layout.addWidget(self.seed_label)
        image_layout.addWidget(self.segment_label)
        image_layout.addStretch()

        return image_layout

    @pyqtSlot()
    def on_foreground(self):
        """Set seed mode to foreground."""
        self.seed_mode = self.graph.foreground
        self._update_button_styles()
        logger.info("Foreground seed mode activated")

    @pyqtSlot()
    def on_background(self):
        """Set seed mode to background."""
        self.seed_mode = self.graph.background
        self._update_button_styles()
        logger.info("Background seed mode activated")

    def _update_button_styles(self):
        """Update button styles based on current seed mode."""
        if self.foreground_button and self.background_button:
            if self.seed_mode == self.graph.foreground:
                self.foreground_button.setStyleSheet("background-color: gray")
                self.background_button.setStyleSheet("background-color: white")
            else:
                self.foreground_button.setStyleSheet("background-color: white")
                self.background_button.setStyleSheet("background-color: gray")

    @pyqtSlot()
    def on_clear(self):
        """Clear all seeds from the image."""
        try:
            self.graph.clear_seeds()
            self._update_image_display()
            logger.info("All seeds cleared")
        except Exception as e:
            logger.error(f"Error clearing seeds: {e}")
            self._show_error("Seed Clearing Error", str(e))

    @pyqtSlot()
    def on_segment(self):
        """Perform image segmentation."""
        try:
            self.graph.create_graph()
            self._update_image_display()
            logger.info("Image segmentation completed")
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            self._show_error("Segmentation Error", str(e))

    @pyqtSlot()
    def on_open(self):
        """Open and load an image file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image",
                "",
                "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)",
            )

            if file_path:
                logger.info(f"Opening image: {file_path}")
                self.graph.load_image(file_path)

                if self.graph.image is not None:
                    self._update_image_display()
                else:
                    raise ValueError("Failed to load image")
        except Exception as e:
            logger.error(f"Image opening error: {e}")
            self._show_error("Image Loading Error", str(e))

    @pyqtSlot()
    def on_save(self):
        """Save the segmented image."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Segmented Image",
                "",
                "Image Files (*.png *.jpg *.bmp);;All Files (*)",
            )

            if file_path:
                self.graph.save_image(file_path)
                logger.info(f"Image saved to: {file_path}")
        except Exception as e:
            logger.error(f"Image saving error: {e}")
            self._show_error("Image Saving Error", str(e))

    def mouse_down(self, event: QMouseEvent):
        """Handle mouse press event for adding seeds."""
        self._add_seed_at_position(event.x(), event.y())

    def mouse_drag(self, event: QMouseEvent):
        """Handle mouse drag event for adding seeds."""
        self._add_seed_at_position(event.x(), event.y())

    def _add_seed_at_position(self, x: int, y: int):
        """Add a seed at the specified position."""
        try:
            # Directly use the raw mouse coordinates without scaling
            self.graph.add_seed(x, y, self.seed_mode)
            self._update_image_display()
        except Exception as e:
            logger.error(f"Error adding seed: {e}")

    def _update_image_display(self):
        """Update seed and segmented image displays."""
        if self.seed_label and self.segment_label:
            seed_image = self.graph.get_image_with_overlay(self.graph.seeds)
            segmented_image = self.graph.get_image_with_overlay(self.graph.segmented)

            seed_pixmap = QPixmap.fromImage(self._convert_cv_to_qimage(seed_image))
            segmented_pixmap = QPixmap.fromImage(
                self._convert_cv_to_qimage(segmented_image)
            )

            # Resize QLabel to match the image size
            self.seed_label.setPixmap(seed_pixmap)
            self.seed_label.setFixedSize(seed_pixmap.size())

            self.segment_label.setPixmap(segmented_pixmap)
            self.segment_label.setFixedSize(segmented_pixmap.size())

    @staticmethod
    def _convert_cv_to_qimage(cv_image: Optional[np.ndarray]) -> QImage:
        """
        Convert OpenCV image to QImage.

        Args:
            cv_image (Optional[np.ndarray]): OpenCV image

        Returns:
            QImage: Converted image or empty QImage
        """
        if cv_image is None:
            return QImage()

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, bytes_per_pixel = cv_image_rgb.shape
        bytes_per_line = width * bytes_per_pixel

        return QImage(
            cv_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

    def _show_error(self, title: str, message: str):
        """
        Show an error message dialog.

        Args:
            title (str): Dialog title
            message (str): Error message
        """
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle(title)
        error_dialog.setText(message)
        error_dialog.exec_()