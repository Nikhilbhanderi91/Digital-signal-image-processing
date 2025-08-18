
CODE:-
import cv2
import numpy as np
from matplotlib import pyplot as plt import os

# Step 1: File paths
# Use your actual path
source_path = "/content/image22 (1).jpeg"
reference_path = "/content/image22 (1).jpeg"  # for demo, using same file as reference


# Step 2: Load images in grayscale
source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE) reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Check loading
if source_image is None:
raise ValueError(f" Could not load source image. Path = {source_path}") if reference_image is None:
raise ValueError(f" Could not load reference image. Path = {reference_path}")


# Step 3: Histogram Equalization
equalized_image = cv2.equalizeHist(source_image)


plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1) plt.title('Original Image')
 
plt.imshow(source_image, cmap='gray') plt.axis('off')

plt.subplot(2, 2, 2) plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray') plt.axis('off')

plt.subplot(2, 2, 3) plt.title('Histogram - Original')
plt.plot(cv2.calcHist([source_image], [0], None, [256], [0, 256])) plt.grid(True)

plt.subplot(2, 2, 4) plt.title('Histogram - Equalized')
plt.plot(cv2.calcHist([equalized_image], [0], None, [256], [0, 256])) plt.grid(True)

plt.tight_layout() plt.show()

# Step 4: Histogram Matching
# Compute histograms (normalized)
src_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
ref_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256]) src_hist /= src_hist.sum()
ref_hist /= ref_hist.sum()
 
# Compute cumulative distribution functions (CDFs) src_cdf = src_hist.cumsum()
ref_cdf = ref_hist.cumsum()


# Mapping from source to reference
mapping = np.interp(src_cdf, ref_cdf, np.arange(256)) matched_image = mapping[source_image].astype(np.uint8)

# Step 5: Show Results plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1) plt.title('Source Image')
plt.imshow(source_image, cmap='gray') plt.axis('off')

plt.subplot(1, 3, 2) plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray') plt.axis('off')

plt.subplot(1, 3, 3) plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray') plt.axis('off')

plt.tight_layout() plt.show()
 

#
# Step 6: Observations #
print("\n Observations:")
print("1. The original image histogram may be narrow (low contrast).")
print("2. Histogram Equalization spreads intensity values across 0-255, improving contrast.") print("3. Histogram Matching adjusts the source so its intensity distribution matches the
reference.")
print("4. Equalization always enhances contrast, while Matching adapts source to look like reference.")

OUTPUT:-
 

OBSERVATION:-
Histogram equalization enhances the overall contrast of an image by spreading out the pixel intensities across the full range, making dark areas appear darker and bright areas brighter, which improves visibility but can sometimes look a bit unnatural. On the other hand, histogram matching adjusts the source image so its brightness and contrast resemble a reference image, giving a result that feels more consistent with the reference’s lighting and style. In short, equalization is great for boosting general contrast, while matching is more controlled and useful when you want one image to visually align with another.

CONCUSTION:-
Histogram equalization is a powerful technique to improve the visibility of images by stretching the contrast, though it may sometimes feel a bit artificial. Histogram matching, on the other hand, is more adaptive, as it shapes the image’s brightness and contrast to follow a reference image, giving a more natural and style-specific result. In conclusion, while equalization is best when the goal is to simply enhance clarity, matching is the preferred choice when we want one image to blend in or look similar to another in terms of lighting and tone.
