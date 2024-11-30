import cv2
import numpy as np
import maxflow

class GraphMaker:
    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000

    def __init__(self):
        self.image = None
        self.graph = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.mask = None
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

    def load_image(self, filename):
        """Load an image and initialize overlays."""
        self.image = cv2.imread(filename)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {filename}")
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

    def add_seed(self, x, y, seed_type):
        """Add a seed for foreground or background."""
        if self.image is None:
            raise ValueError("Please load an image before adding seeds.")
        if seed_type == self.background:
            if (x, y) not in self.background_seeds:
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)
        elif seed_type == self.foreground:
            if (x, y) not in self.foreground_seeds:
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)

    def clear_seeds(self):
        """Clear all seeds."""
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        """Get the current overlay."""
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        return self.segment_overlay

    def get_image_with_overlay(self, overlay_type):
        """Combine the base image with the selected overlay."""
        if self.image is None:
            raise ValueError("Image is not loaded. Please load an image first.")
        overlay = self.seed_overlay if overlay_type == self.seeds else self.segment_overlay
        return cv2.addWeighted(self.image, 0.9, overlay, 0.4, 0.1)

    def create_graph(self):
        """Create and segment the graph."""
        if not self.background_seeds or not self.foreground_seeds:
            raise ValueError("Please enter at least one foreground and one background seed.")
        self.find_averages()
        self.populate_graph()
        self.cut_graph()

    def find_averages(self):
        """Calculate averages for foreground and background regions."""
        self.graph = np.full((self.image.shape[0], self.image.shape[1]), self.default)
        for x, y in self.background_seeds:
            self.graph[y, x] = 0
        for x, y in self.foreground_seeds:
            self.graph[y, x] = 1

    def populate_graph(self):
        """Create nodes and edges for the graph."""
        self.nodes = []
        self.edges = []

        for (y, x), value in np.ndenumerate(self.graph):
            node_id = self.get_node_num(x, y, self.image.shape)
            if value == 0.0:
                self.nodes.append((node_id, self.MAXIMUM, 0))
            elif value == 1.0:
                self.nodes.append((node_id, 0, self.MAXIMUM))
            else:
                self.nodes.append((node_id, 0, 0))

        for (y, x), _ in np.ndenumerate(self.graph):
            if y < self.graph.shape[0] - 1:
                g = 1 / (1 + np.sum((self.image[y, x] - self.image[y+1, x]) ** 2))
                self.edges.append((self.get_node_num(x, y, self.image.shape),
                                   self.get_node_num(x, y+1, self.image.shape), g))
            if x < self.graph.shape[1] - 1:
                g = 1 / (1 + np.sum((self.image[y, x] - self.image[y, x+1]) ** 2))
                self.edges.append((self.get_node_num(x, y, self.image.shape),
                                   self.get_node_num(x+1, y, self.image.shape), g))

    def cut_graph(self):
        """Perform graph cut using the maxflow algorithm."""
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float]()
        node_ids = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(node_ids[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        g.maxflow()

        for i, node in enumerate(self.nodes):
            if g.get_segment(i) == 1:
                x, y = self.get_xy(node[0], self.image.shape)
                self.segment_overlay[y, x] = (255, 0, 255)
                self.mask[y, x] = True

    def swap_overlay(self, overlay_type):
        """Switch between seed and segmentation overlays."""
        self.current_overlay = overlay_type

    def save_image(self, filename):
        """Save the segmented portion of the image."""
        if self.mask is None:
            raise ValueError("Please segment the image before saving.")
        segmented = np.zeros_like(self.image)
        np.copyto(segmented, self.image, where=self.mask)
        cv2.imwrite(filename, segmented)

    @staticmethod
    def get_node_num(x, y, array_shape):
        """Convert pixel coordinates to a node number."""
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(node_num, array_shape):
        """Convert a node number back to pixel coordinates."""
        return node_num % array_shape[1], node_num // array_shape[1]

