import cv2
import os
import numpy as np

fileName = "06052.tif"
file_path = "../../fullScrollDataTest/" + fileName

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
bright_color = np.array([255, 0, 0], dtype=np.uint8)

# Apply the bright color to the different pixels using the mask
colored_diff = np.where(mask[..., np.newaxis] == 1, bright_color, diff_image)

# Save the resulting image
outputFilePath = "../../comparisonScrollData/comp" + fileName
cv2.imwrite(
    outputFilePath,
    cv2.cvtColor(colored_diff, cv2.COLOR_RGB2BGR),
)
