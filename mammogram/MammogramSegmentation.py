import pywt
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from typing import Optional

class MammogramSegmentation:
    """
    Performs mammogram segmentation using wavelet transform and k-means clustering.
    """

    def __init__(self, image: np.ndarray):
        """
        Initializes the MammogramSegmentation class.

        Args:
            image (np.ndarray): The input mammogram image as a NumPy array.  Should be grayscale.
        
        Raises:
            TypeError: If input image is not a NumPy array.
            ValueError: If input image is not 2-dimensional (grayscale).

        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if image.ndim != 2:
            raise ValueError("Input image must be grayscale (2-dimensional).")
        self.image = image
        self.segmented_image: Optional[np.ndarray] = None


    def wavelet_transform(self, wavelet: str = "haar") -> None:
        """
        Applies a 2D discrete wavelet transform to the image, zeros out the LL band, and performs inverse transform.

        Args:
            wavelet (str, optional): The type of wavelet to use. Defaults to "haar".
        """
        try:
            coeffs = pywt.dwt2(self.image, wavelet)
            LL, (LH, HL, HH) = coeffs
            LL *= 0  # Zero out the low-frequency component
            self.transformed_image = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
            self.final_image = self.image.astype(np.float32) + self.transformed_image
        except Exception as e:
            print(f"Error during wavelet transform: {e}")
            self.final_image = None # Indicate failure


    def kmeans_clustering(self, n_clusters: int = 10) -> None:
        """
        Performs k-means clustering on the wavelet-transformed image.

        Args:
            n_clusters (int, optional): The number of clusters. Defaults to 10.
        
        Raises:
            ValueError: If wavelet transform hasn't been performed.
        """
        if self.final_image is None:
            raise ValueError("Wavelet transform must be performed first.")

        try:
            reshaped_image = self.final_image.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=13)  #n_init increased for robustness
            kmeans.fit(reshaped_image)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            self.segmented_image = centers[labels.flatten()].reshape(self.image.shape).astype(np.uint8)
            _, self.segmented_image = cv.threshold(self.segmented_image, 200, 255, cv.THRESH_BINARY)
        except Exception as e:
            print(f"Error during k-means clustering: {e}")
            self.segmented_image = None # Indicate failure


    def get_segmented_image(self) -> Optional[np.ndarray]:
      """Returns the segmented image. Returns None if segmentation failed."""
      return self.segmented_image