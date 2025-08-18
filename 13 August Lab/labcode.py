# ===== Import required libraries =====
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ==============================
# 1. Histogram Equalization
# ==============================

# Load the image in grayscale
image_path = '/Users/nikhilbhanderi/Downloads/ex1_4.png'  # Change path if needed
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    raise FileNotFoundError(f"Image not found at {'/Users/nikhilbhanderi/Downloads/ex1_4.png'}")

# Calculate histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot histogram
plt.figure(figsize=(8, 6))
plt.title('Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Display original & equalized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')
plt.show()

# Equalized histogram
equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(equalized_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# ==============================
# 2. Histogram Matching
# ==============================

# Load source and reference images
source_path = '/content/IMG_20230915_135931789.jpg'       # Change to actual path
reference_path = '/content/matched_image.jpg'            # Change to actual path

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

if source_image is None or reference_image is None:
    raise FileNotFoundError("Source or Reference image not found!")

# Calculate histograms
source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

# Normalize histograms
source_hist /= source_hist.sum()
reference_hist /= reference_hist.sum()

# Calculate CDFs
source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

# Histogram matching mapping
mapping = np.interp(source_cdf, reference_cdf, range(256))
matched_image = mapping[source_image]
matched_image = matched_image.astype(np.uint8)

# Display images
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')
plt.show()
