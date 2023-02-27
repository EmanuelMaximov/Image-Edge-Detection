## Description
I made cv2.filter2D (from opencv library) manually - created default kernel in the entered dimensions
and then applied the kernel to the image and applied mirroring algorithm  for the borders

Filter options:
1. Smoothing - Performs the convolution on colored Image
2. Edge detection - Performs the convolution on grayscale Image with Laplacian of Gaussian (LoG) & Sobel filters



## How to run the code:
- Python 3.10
- Input: Image file name (relative pathname)
- Output: 3 images: 1. Original Image 2. Filtered Image 3. Difference Image

In terminal: 
``` python main.py "Image.jpg" ```
