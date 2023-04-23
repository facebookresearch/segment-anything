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

# sam_checkpoint = "segment-anything\sam_vit_l_0b3195.pth"
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
filePath = "../../fullScrollDataTest/06052.tif"


# coco_rle visualization
def visualize_rle_mask(image, rle_mask, alpha=0.5):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

    # Make sure the binary mask dimensions match the image dimensions
    binary_mask = cv2.resize(
        binary_mask.astype(np.uint8), (image.shape[1], image.shape[0])
    )

    # Overlay the binary mask on the image
    masked_image = image.copy()
    masked_image[binary_mask == 1] = (
        masked_image[binary_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)

    return masked_image


# TODO - this is not working, visualize multiple masks
def visualize_rle_masks(image, rle_masks, alpha=0.5):
    # Create a copy of the image to overlay the masks
    masked_image = image.copy()

    for rle_mask in rle_masks:
        # Decode the RLE mask into a binary mask
        binary_mask = coco_mask.decode(rle_mask)

        # Overlay the binary mask on the image
        masked_image[binary_mask == 1] = (
            masked_image[binary_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        ).astype(np.uint8)

    return masked_image


def show_image(image):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def scale_rle_mask(rle_mask, scale_factor):
    # Decode the RLE mask into a binary mask
    binary_mask = coco_mask.decode(rle_mask)

    # Resize the binary mask
    new_height = int(binary_mask.shape[0] * scale_factor)
    new_width = int(binary_mask.shape[1] * scale_factor)
    resized_binary_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Encode the resized binary mask back into an RLE mask
    scaled_rle_mask = coco_mask.encode(np.asfortranarray(resized_binary_mask))
    return scaled_rle_mask


def downsample_image(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Use cv2.resize() to downsample the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam, output_mode="coco_rle"
)  # ,output_mode="coco_rle
image = cv2.imread(filePath)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


scale_factor = (
    0.1  # Reduce the dimensions of the original size so batch can fit in GPU memory
)
downsampled_image = downsample_image(image, scale_factor)
masks = mask_generator.generate(downsampled_image)

print(len(masks))
print(masks[0].keys())
# print(masks)

# rle_image = visualize_rle_mask(downsampled_image, masks[0]["segmentation"])
# show_image(rle_image)

scaled_rle_mask = scale_rle_mask(masks[0]["segmentation"], 1 / scale_factor)
rle_image = visualize_rle_mask(image, scaled_rle_mask)
show_image(rle_image)
