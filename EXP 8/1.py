



CODE:-
import cv2
import numpy as np
from matplotlib import pyplot as plt


# Step 1: Paths of uploaded images (all 5) image_paths = [
"/content/ex1_1.png", "/content/ex1_2.png", "/content/ex1_3.png", "/content/ex1_4.png", "/content/ex1_5.png"
]


# Step 2: Function for histogram equalization def histogram_equalization(image):
return cv2.equalizeHist(image)


# Step 3: Function for histogram matching def histogram_matching(source, reference):
# Compute histograms (normalized)
src_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
ref_hist = cv2.calcHist([reference], [0], None, [256], [0, 256]) src_hist /= src_hist.sum()
ref_hist /= ref_hist.sum()


# Compute cumulative distribution functions (CDFs)
 
src_cdf = src_hist.cumsum() ref_cdf = ref_hist.cumsum()

# Mapping from source to reference
mapping = np.interp(src_cdf, ref_cdf, np.arange(256)) matched = mapping[source].astype(np.uint8)
return matched


# Step 4: Use first image as reference
ref_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)


# Step 5: Process each image
for i, path in enumerate(image_paths): # Load as grayscale
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if img is None:
print(f" Could not load image {path}") continue

# Histogram Equalization
eq_img = histogram_equalization(img)


# Histogram Matching with reference image matched_img = histogram_matching(img, ref_img)

# Show results plt.figure(figsize=(12, 6))
 
plt.subplot(1, 3, 1) plt.title(f'Original {i+1}') plt.imshow(img, cmap='gray') plt.axis('off')

plt.subplot(1, 3, 2) plt.title(f'Equalized {i+1}') plt.imshow(eq_img, cmap='gray') plt.axis('off')

plt.subplot(1, 3, 3) plt.title(f'Matched {i+1}')
plt.imshow(matched_img, cmap='gray') plt.axis('off')

plt.tight_layout() plt.show()

# Step 6: Observations print("\n Observations:")
print("1. Histogram Equalization improves overall contrast by redistributing intensity values.") print("2. Histogram Matching modifies the input so it looks more like the reference image.")
print("3. Equalization is good for general enhancement, while Matching is useful when you want two images to have a similar tone distribution.")
 
OUTPUT:-
 
























OBSERVATION:-
When we applied histogram equalization to the images, the contrast clearly improved – darker areas became more visible and brighter areas stood out more. However, in some cases, equalization gave the picture an unnatural “over-sharpened” look because it simply stretched the intensity values across the full range.
On the other hand, histogram matching produced results that were closer to the style of the reference image. Instead of just boosting contrast blindly, it adjusted the tones so that each image shared a similar brightness and contrast pattern with the reference. This made the output look more balanced and natural compared to equalization.

CONCUSTION:-
In simple words, histogram equalization is great when we just want to enhance visibility and improve overall contrast, but it doesn’t always preserve the natural look of the image. Histogram matching is more controlled and context-aware, as it adapts the image intensities to resemble a chosen reference image.
