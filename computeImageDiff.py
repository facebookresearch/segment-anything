import cv2
import os

fileName = "06052.tif"
file_path = "../../fullScrollDataTest/" + fileName

# Load the images
image1 = cv2.imread(file_path)
image2 = cv2.imread("../../losslesslyCompressedScrollData/c" + fileName)
# Ensure the images have the same size and number of channels
if image1.shape != image2.shape:
    raise ValueError("The two images must have the same size and number of channels.")

# Compute the absolute difference between the two images
diff_image = cv2.absdiff(image1, image2) * 100

# Save the resulting image
outputFilePath = "../../comparisonScrollData/comp" + fileName
cv2.imwrite(
    outputFilePath,
    cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR),
)
