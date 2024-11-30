import cv2
import numpy as np
import maxflow
import logging
from typing import List, Tuple, Optional, Union


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Graph:
    """
    A class for performing graph-based image segmentation using max-flow/min-cut algorithm.
    
    Attributes:
        foreground (int): Constant representing foreground seeds
        background (int): Constant representing background seeds
    """
    
    # Class-level constants
    foreground: int = 1
    background: int = 0
    
    seeds: int = 0
    segmented: int = 1
    
    DEFAULT_SEED_VALUE: float = 0.5
    MAXIMUM_NODE_WEIGHT: int = 1_000_000_000
    
    SEED_COLORS = {
        background: (0, 0, 255),   # Blue for background
        foreground: (0, 255, 0)    # Green for foreground
    }
    
    def __init__(self):
        """Initialize Graph with default attributes."""
        self.image: Optional[np.ndarray] = None
        self.graph: Optional[np.ndarray] = None
        self.overlay: Optional[np.ndarray] = None
        self.seed_overlay: Optional[np.ndarray] = None
        self.segment_overlay: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        
        self.background_seeds: List[Tuple[int, int]] = []
        self.foreground_seeds: List[Tuple[int, int]] = []
        
        self.current_overlay: int = self.seeds
        
        # Initialize empty numpy arrays
        self.background_average: np.ndarray = np.zeros(3)
        self.foreground_average: np.ndarray = np.zeros(3)
        
        self.nodes: List[Tuple[int, float, float]] = []
        self.edges: List[Tuple[int, int, float]] = []
    
    def load_image(self, filename: str) -> None:
        """
        Load an image and initialize overlays.
        
        Args:
            filename (str): Path to the image file
        
        Raises:
            IOError: If the image cannot be read
        """
        try:
            self.image = cv2.imread(filename)
            
            if self.image is None:
                raise IOError(f"Unable to read image file: {filename}")
            
            # Initialize overlays
            self.seed_overlay = np.zeros_like(self.image)
            self.segment_overlay = np.zeros_like(self.image)
            self.mask = None
            
            logger.info(f"Image loaded successfully: {filename}")
        
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def add_seed(self, x: int, y: int, seed_type: int) -> None:
        """
        Add a seed point to the image.
        
        Args:
            x (int): X-coordinate of the seed
            y (int): Y-coordinate of the seed
            seed_type (int): Type of seed (foreground or background)
        """
        if self.image is None:
            raise ValueError("Please load an image before adding seeds.")
        
        # Prevent duplicate seeds
        seed_coords = (x, y)
        seed_list = (self.background_seeds if seed_type == self.background 
                     else self.foreground_seeds)
        
        if seed_coords not in seed_list:
            seed_list.append(seed_coords)
            color = self.SEED_COLORS[seed_type]
            cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), color, -1)
    
    def clear_seeds(self) -> None:
        """Clear all seeds and reset seed overlay."""
        self.background_seeds.clear()
        self.foreground_seeds.clear()
        
        if self.seed_overlay is not None:
            self.seed_overlay.fill(0)
    
    # Rest of the methods remain similar to the original implementation...
    
    def get_overlay(self) -> np.ndarray:
        """
        Get the current overlay.
        
        Returns:
            np.ndarray: Current overlay image
        """
        return (self.seed_overlay if self.current_overlay == self.seeds 
                else self.segment_overlay)

    def get_image_with_overlay(self, overlay_type: int) -> np.ndarray:
        """
        Combine the base image with the selected overlay.
        
        Args:
            overlay_type (int): Type of overlay to apply (seeds or segmented)
        
        Returns:
            np.ndarray: Image with overlay applied
        
        Raises:
            ValueError: If no image is loaded
            TypeError: If overlay type is invalid
        """
        if self.image is None:
            raise ValueError("Image is not loaded. Please load an image first.")
        
        if overlay_type not in [self.seeds, self.segmented]:
            raise TypeError(f"Invalid overlay type: {overlay_type}")
        
        overlay = (self.seed_overlay if overlay_type == self.seeds 
                   else self.segment_overlay)
        
        return cv2.addWeighted(
            self.image, 0.9,  # Source image and its weight
            overlay, 0.4,     # Overlay and its weight
            0.1               # Gamma correction
        )

    def create_graph(self, smoothness_factor: float = 1.0) -> None:
        """
        Create and segment the graph.
        
        Args:
            smoothness_factor (float, optional): Factor to adjust edge weights. 
                                                 Defaults to 1.0.
        
        Raises:
            ValueError: If insufficient seeds are provided
        """
        if not self.background_seeds or not self.foreground_seeds:
            raise ValueError(
                "Please enter at least one foreground and one background seed."
            )
        
        try:
            self._find_seed_averages()
            self._populate_graph_nodes_and_edges(smoothness_factor)
            self._apply_max_flow_cut()
        except Exception as e:
            logging.error(f"Graph creation failed: {e}")
            raise

    def _find_seed_averages(self) -> None:
        """
        Initialize graph with seed values.
        Marks background and foreground seeds in the graph.
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Use class default value
        self.graph = np.full(
            (self.image.shape[0], self.image.shape[1]), 
            self.DEFAULT_SEED_VALUE
        )
        
        # Mark background and foreground seeds
        for x, y in self.background_seeds:
            self.graph[y, x] = 0.0
        
        for x, y in self.foreground_seeds:
            self.graph[y, x] = 1.0

    def _populate_graph_nodes_and_edges(self, smoothness: float = 1.0) -> None:
        """
        Create nodes and edges for graph-cut segmentation.
        
        Args:
            smoothness (float): Factor to adjust edge weights
        """
        self.nodes: List[Tuple[int, float, float]] = []
        self.edges: List[Tuple[int, int, float]] = []

        # Create nodes based on seed values
        for (y, x), value in np.ndenumerate(self.graph):
            node_id = self.get_node_num(x, y, self.image.shape)
            
            if value == 0.0:  # Background seed
                self.nodes.append((node_id, self.MAXIMUM_NODE_WEIGHT, 0))
            elif value == 1.0:  # Foreground seed
                self.nodes.append((node_id, 0, self.MAXIMUM_NODE_WEIGHT))
            else:  # Undefined region
                self.nodes.append((node_id, 0, 0))

        # Create edges between neighboring pixels
        for (y, x), _ in np.ndenumerate(self.graph):
            # Vertical edge
            if y < self.graph.shape[0] - 1:
                edge_weight = self._calculate_edge_weight(
                    self.image[y, x], 
                    self.image[y+1, x], 
                    smoothness
                )
                self.edges.append((
                    self.get_node_num(x, y, self.image.shape),
                    self.get_node_num(x, y+1, self.image.shape), 
                    edge_weight
                ))
            
            # Horizontal edge
            if x < self.graph.shape[1] - 1:
                edge_weight = self._calculate_edge_weight(
                    self.image[y, x], 
                    self.image[y, x+1], 
                    smoothness
                )
                self.edges.append((
                    self.get_node_num(x, y, self.image.shape),
                    self.get_node_num(x+1, y, self.image.shape), 
                    edge_weight
                ))

    @staticmethod
    def _calculate_edge_weight(
        pixel1: np.ndarray, 
        pixel2: np.ndarray, 
        smoothness: float
    ) -> float:
        """
        Calculate edge weight between two pixels.
        
        Args:
            pixel1 (np.ndarray): First pixel
            pixel2 (np.ndarray): Second pixel
            smoothness (float): Smoothness factor
        
        Returns:
            float: Calculated edge weight
        """
        # Smaller difference = higher weight (more likely to be in same segment)
        pixel_diff = np.sum((pixel1 - pixel2) ** 2)
        return smoothness / (1 + pixel_diff)

    def _apply_max_flow_cut(self) -> None:
        """
        Apply max-flow/min-cut algorithm to segment the image.
        """
        # Initialize overlay and mask
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = np.zeros_like(self.image, dtype=bool)

        # Create graph
        graph = maxflow.Graph[float]()
        node_ids = graph.add_nodes(len(self.nodes))

        # Add terminal edges
        for node in self.nodes:
            graph.add_tedge(node_ids[node[0]], node[1], node[2])

        # Add inter-pixel edges
        for edge in self.edges:
            graph.add_edge(edge[0], edge[1], edge[2], edge[2])

        # Compute max flow
        graph.maxflow()

        # Mark segmented regions
        for i, node in enumerate(self.nodes):
            if graph.get_segment(i) == 1:
                x, y = self.get_xy(node[0], self.image.shape)
                self.segment_overlay[y, x] = (255, 0, 255)  # Magenta
                self.mask[y, x] = True

    def swap_overlay(self, overlay_type: int) -> None:
        """
        Switch between seed and segmentation overlays.
        
        Args:
            overlay_type (int): Type of overlay to display
        
        Raises:
            ValueError: If invalid overlay type is provided
        """
        if overlay_type not in [self.seeds, self.segmented]:
            raise ValueError(f"Invalid overlay type: {overlay_type}")
        
        self.current_overlay = overlay_type

    def save_image(self, filename: str) -> None:
        """
        Save the segmented image.
        
        Args:
            filename (str): Output filename
        
        Raises:
            ValueError: If image hasn't been segmented
        """
        if self.mask is None:
            raise ValueError("Please segment the image before saving.")
        
        segmented = np.zeros_like(self.image)
        np.copyto(segmented, self.image, where=self.mask)
        
        cv2.imwrite(filename, segmented)

    @staticmethod
    def get_node_num(x: int, y: int, array_shape: Tuple[int, int, int]) -> int:
        """
        Convert pixel coordinates to a linear node number.
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
            array_shape (Tuple[int, int, int]): Shape of the image array
        
        Returns:
            int: Linearized node number
        """
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(node_num: int, array_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Convert a linear node number back to pixel coordinates.
        
        Args:
            node_num (int): Linear node number
            array_shape (Tuple[int, int, int]): Shape of the image array
        
        Returns:
            Tuple[int, int]: (x, y) pixel coordinates
        """
        return node_num % array_shape[1], node_num // array_shape[1]