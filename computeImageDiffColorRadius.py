import cv2
import os
import numpy as np

fileName = "06052.tif"  # file to compare before and after lossless compression
file_path = "../../fullScrollData/" + fileName

# Load the images
image1 = cv2.imread(file_path)
image2 = cv2.imread("../../losslesslyCompressedScrollData/c" + fileName)
# Ensure the images have the same size and number of channels
if image1.shape != image2.shape:
    raise ValueError("The two images must have the same size and number of channels.")

# Compute the absolute difference between the two images
diff_image = cv2.absdiff(image1, image2)

# Set a threshold value for determining if pixels are different
threshold_value = 1

# Create a binary mask where pixel differences are greater than the threshold value
mask = np.where(np.all(diff_image > threshold_value, axis=-1), 1, 0)

# Create an array with the bright color (e.g., red) to apply to the different pixels
bright_color = np.array([205, 0, 0], dtype=np.uint8)

radius = 5

# Create a circular structuring element for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

# Dilate the binary mask to create a radius around the different pixels
dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel)

# Apply the bright color to the dilated mask
colored_diff = np.where(dilated_mask[..., np.newaxis] == 1, bright_color, diff_image)

# Save the resulting image
outputFilePath = "../../comparisonScrollData/comp" + fileName
cv2.imwrite(
    outputFilePath,
    cv2.cvtColor(colored_diff, cv2.COLOR_RGB2BGR),
)
