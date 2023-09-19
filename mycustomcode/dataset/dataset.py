
from torch.utils.data import Dataset
import numpy as np



def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
      self.dataset = dataset
      self.processor = processor

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      item = self.dataset[idx]
      image = item["image"]
      ground_truth_mask = np.array(item["label"])

      # get bounding box prompt
      prompt = get_bounding_box(ground_truth_mask)

      # prepare image and prompt for the model
      inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

      # remove batch dimension which the processor adds by default
      inputs = {k:v.squeeze(0) for k,v in inputs.items()}

      # add ground truth segmentation
      inputs["ground_truth_mask"] = ground_truth_mask

      return inputs


if __name__ == "__main__":
    dataset = load_dataset("nielsr/breast-cancer", split="train")
    sample =  dataset[0]
    fig, axes = plt.subplots()

    axes.imshow(np.array(sample['image']))
    ground_truth_seg = np.array(sample["label"])
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"Ground truth mask")
    axes.axis("off")
