import sys
sys.path.append("..")
import numpy as np

from interactive_predictor import build_sam
import torch


sam = build_sam()

#dummy_image = torch.randint(low=0, high=255, size=(1500, 2250, 3), dtype=torch.uint8)
dummy_image = np.random.randint(low=0, high=255, size=(1500, 2250, 3), dtype=np.uint8)

dummy_x_coords = torch.randint(low=0, high=2250, size=(1,3,1), dtype=torch.int)
dummy_y_coords = torch.randint(low=0, high=1500, size=(1,3,1), dtype=torch.int)
dummy_coords = torch.cat([dummy_x_coords, dummy_y_coords], dim=-1)

dummy_labels = torch.tensor([[1,-1,0]], dtype=torch.int64)

print("Setting image.")
sam.set_image(dummy_image)
print("Resulting features shape:", sam.features.shape)

print("Predicting round one...")

masks, iou_preds, low_res = sam.predict(dummy_coords, dummy_labels, None, True)

print("Masks shape:", masks.shape)
print("IoU pred shape:", iou_preds.shape)
print("Low res masks shape:", low_res.shape)

print("Predicting round two...")

masks2, iou_preds2, low_res2 = sam.predict(dummy_coords, dummy_labels, low_res[:,:1], False)

print("Masks shape:", masks2.shape)
print("IoU pred shape:", iou_preds2.shape)
print("Low res masks shape:", low_res2.shape)





