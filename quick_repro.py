from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

img = np.array(Image.open("samples/dog_man_1_crop.jpg").convert("RGB"))

predictor.set_image(img)
masks, _, _ = predictor.predict(np.array([[250, 200]]), np.array([1.0]))

fig, axes = plt.subplots(1, 4, figsize=(4*5, 5))

axes[0].imshow(img)
axes[1].imshow(masks[0])
axes[2].imshow(masks[1])
axes[3].imshow(masks[2])
plt.tight_layout()
plt.show()
print("here")