import cv2
import numpy as np
import logging
from typing import Optional

from GraphMaker import GraphMaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CutUI:
    """
    An OpenCV-based user interface for graph-based image segmentation.
    
    Manages interactive image segmentation using mouse interactions.
    """
    
    def __init__(self, filename: Optional[str] = None):
        """
        Initialize the UI with an optional image.
        
        Args:
            filename (Optional[str]): Path to the input image
        """
        self.graph_maker = GraphMaker()
        
        if filename:
            try:
                self.graph_maker.load_image(filename)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                raise
        
        self.display_image: np.ndarray = np.array(self.graph_maker.image) if self.graph_maker.image is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Window configuration
        self.window: str = "Graph Cut Segmentation"
        self.mode: int = self.graph_maker.foreground
        self.started_click: bool = False
    
    def run(self):
        """Run the interactive segmentation interface."""
        if self.graph_maker.image is None:
            logger.warning("No image loaded. Please load an image first.")
            return
        
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._draw_line)
        
        try:
            while True:
                display = cv2.addWeighted(
                    self.display_image, 0.9, 
                    self.graph_maker.get_overlay(), 0.4, 0.1
                )
                cv2.imshow(self.window, display)
                
                key = cv2.waitKey(20) & 0xFF
                if self._handle_key_events(key):
                    break
        
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
        
        finally:
            cv2.destroyAllWindows()
    
    def _handle_key_events(self, key: int) -> bool:
        """
        Handle keyboard events during segmentation.
        
        Args:
            key (int): Pressed key code
        
        Returns:
            bool: Whether to exit the application
        """
        if key == 27:  # ESC key
            return True
        
        elif key == ord('c'):  # Clear seeds
            self.graph_maker.clear_seeds()
        
        elif key == ord('g'):  # Create graph and segment
            try:
                self.graph_maker.create_graph()
                self.graph_maker.swap_overlay(self.graph_maker.segmented)
            except Exception as e:
                logger.error(f"Graph creation error: {e}")
        
        elif key == ord('t'):  # Toggle seed mode
            self.mode = 1 - self.mode
            self.graph_maker.swap_overlay(self.graph_maker.seeds)
        
        return False
    
    def _draw_line(self, event: int, x: int, y: int, flags: int, param: Optional[object] = None):
        """
        Handle mouse drawing events for adding seeds.
        
        Args:
            event (int): Mouse event type
            x (int): X-coordinate
            y (int): Y-coordinate
            flags (int): Event flags
            param (Optional[object]): Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.started_click = True
            self._add_seed(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.started_click = False
        
        elif event == cv2.EVENT_MOUSEMOVE and self.started_click:
            self._add_seed(x, y)
    
    def _add_seed(self, x: int, y: int):
        """
        Add a seed point to the image.
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
        """
        self.graph_maker.add_seed(x - 1, y - 1, self.mode)

def main():
    """Main entry point for the application."""
    import sys
    
    if len(sys.argv) < 2:
        logger.error("Please provide an image filename.")
        sys.exit(1)
    
    cut_ui = CutUI(sys.argv[1])
    cut_ui.run()

if __name__ == "__main__":
    main()