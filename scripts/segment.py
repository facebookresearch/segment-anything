import cv2
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def replace_numpy_with_list(item):
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            item[key] = value.tolist()
        elif isinstance(value, dict):
            item[key] = replace_numpy_with_list(value)
    return item


sam_checkpoint = "./Grounded-Segment-Anything-main/sam_vit_b_01ec64.pth" # model checkpoint
model_type = "vit_b" # model type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")

# create the output directory if it doesn't exist
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# path to the directory containing images to be segmented
image_dir = "./Grounded-Segment-Anything-main/segment_anything/inputs"

# iterate over each image in the directory
for i, image_filename in enumerate(os.listdir(image_dir)):
    if not image_filename.endswith((".png", ".jpg", ".jpeg","JPG")):
        continue

    # get the full path to the image file
    image_path = os.path.join(image_dir, image_filename)

    image = cv2.imread(f"./Grounded-Segment-Anything-main/segment_anything/inputs/{image_filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # generate the masks
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    for d in masks:
        d['segmentation'] = d['segmentation'].astype(int)

    masks = [replace_numpy_with_list(item) for item in masks]
    masks = {"masks" : masks}
    print(masks)
    json_ = json.dumps(masks)
    output = json.loads(json_)
    output_file = os.path.join(os.getcwd(), "output.json")

    with open(output_file, "w") as f:
        json.dump(masks, f)

    # path to the output directory
    output_dir = "./Grounded-Segment-Anything-main/segment_anything/outputs"

    # create a filename for the output file
    output_filename = f"{i+1}_{image_filename}"

    # Save the image to disk
    output_path = os.path.join(output_dir, f"{output_filename}")
    plt.savefig(output_path)

    print("Output file saved to:", output_path)



