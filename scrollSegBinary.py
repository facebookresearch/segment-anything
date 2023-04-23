from time import sleep
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as coco_mask
import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
torch.cuda.empty_cache()

"""
Development Script, scrollSegRLESeqRange.py is the most up to date script
"""

# sam_checkpoint = "segment-anything\sam_vit_l_0b3195.pth"
sam_checkpoint = "segment-anything\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
filePath = "../fullScrollDataTest/06052.tif"


# binary mask visualization
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


# coco_rle visualization
def visualize_rle_mask(image, rle_mask, alpha=0.5):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

    # Overlay the binary mask on the image
    masked_image = image.copy()
    masked_image[binary_mask == 1] = (
        masked_image[binary_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    return masked_image


def scale_binary_masks(binary_mask, scale_factor):
    new_height = int(binary_mask.shape[0] * scale_factor)
    new_width = int(binary_mask.shape[1] * scale_factor)
    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized_binary_mask


def downsample_image(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Use cv2.resize() to downsample the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam)  # ,output_mode="coco_rle
image = cv2.imread(filePath)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# plt.axis("off")
# plt.show()

scale_factor = 0.1  # Reduce the dimensions to 50% of the original size
downsampled_image = downsample_image(image, scale_factor)

# plt.figure(figsize=(20, 20))
# plt.imshow(downsampled_image)
# plt.axis("off")
# plt.show()

masks = mask_generator.generate(downsampled_image)

print(len(masks))
print(masks[0].keys())
# print(masks)

plt.figure(figsize=(20, 20))
plt.imshow(downsampled_image)
show_anns(masks)
plt.axis("off")
plt.show()

scaled_masks = scale_binary_masks(masks, 1 / scale_factor)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(scaled_masks)
plt.axis("off")
plt.show()

# rle_image = visualize_rle_mask(downsampled_image, masks[0]["segmentation"])

# plt.figure(figsize=(20, 20))
# plt.imshow(rle_image)
# plt.axis("off")
# plt.show()

# rle_image = visualize_rle_mask(image, masks[0]["segmentation"])

# plt.figure(figsize=(20, 20))
# plt.imshow(rle_image)
# plt.axis("off")
# plt.show()
