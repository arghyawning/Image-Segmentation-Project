# Image Segmentation (and Medical Applications)

Project for the Digital Image Processing course at IIIT Hyderabad (Monsoon 2024).

Team: Sanika Damle, Arghya Roy

## Central Project Idea

Image Segmentation using

1. Graph Cuts
2. Wavelet Transforms

## Description

### General Idea

Image segmentation is a task where an image is divided into distinct regions or segments to identify objects, boundaries or features in the image. This project involves implementing and evaluating two distinct methods for image segmentation: Graph Cuts and Wavelet Transforms. 

**Graph Cuts:** 

- This algorithm involves formulating the segmentation problem as a graph, with each pixel as a node connected to neighboring pixels with edges weighted according to pixel similarity.
- A min-cut/max-flow algorithm is used to find the optimal cut between these pixels.
- This cut divides the graph into disjoint subsets, corresponding to the segmented object and the background.

**Wavelet Transforms:**

- This method is more specific to image segmentation for medical applications.
- The image is decomposed into multiple levels of resolution using a wavelet transform.
- This breaks it down into sub-bands representing different levels of detail.
- Significant features such as edges and textures are extracted from the wavelet coefficients.
- Thresholding techniques are used to isolate features, followed by K-means to group similar pixels.

### Applications

**Brain Tumor Segmentation**

- Undecimated wavelet transforms are combined with Gabor wavelets.
- The undecimated wavelet transforms retain spatial resolution, which make it more effective in detecting tumor boundaries.
- Gabor wavelets enhance the texture and edge information.
- K-means is used for segmentation.

**Mammogram Segmentation**

- Wavelet transforms are used to capture high and low frequency components.
- These high frequency components represent edges and fine details and are crucial for identifying anomalies.
- K-means is used for segmentation to find different tissue types, and find regions that indicate the presence of tumors.

### Ablations

- Preprocessing the image using various transforms/filters to get better segmentation results (example: unsharp masking, high-boost filtering).
- Comparing the effectiveness of both the methods on MRI scans.

## References

1. Boykov, Yuri, and Gareth Funka-Lea. "Graph cuts and efficient ND image segmentation." International journal of computer vision 70.2 (2006): 109-131.
https://www.csd.uwo.ca/~yboykov/Papers/ijcv06.pdf
2. Mirajkar, Gayatri, and Balaji Barbadekar. "Automatic segmentation of brain tumors from MR images using undecimated wavelet transform and gabor wavelets." 2010 17th IEEE International Conference on Electronics, Circuits and Systems. IEEE, 2010.
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5724609
3. Dalmiya, Shruti, Avijit Dasgupta, and Soumya Kanti Datta. "Application of wavelet based K-means algorithm in mammogram segmentation." International Journal of Computer Applications 52.15 (2012).
https://www.ijcaonline.org/archives/volume52/number15/8276-1883/

<br>

---