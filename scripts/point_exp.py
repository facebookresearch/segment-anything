import os
from tqdm import tqdm
from scripts.render import Renderer
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from utils import \
    make_nested_dir, get_data_paths, GROUP3, \
    load_file_npz, load_img

from point_utils import make_point_from_mask


def load_model() -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor

def point_experiment():
    data_path, mask_path = get_data_paths(GROUP3)
    chosen_class = 1
    predictor = load_model()

    prefix = os.path.dirname(data_path[0])
    out_dir = os.path.join(f"{prefix}-fig-class-{chosen_class}")
    make_nested_dir(out_dir)
    point_coords = None
    point_labels = None
    r = Renderer()

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)

        # Only calculate 1
        if point_coords is None or point_labels is None:
            point_coords, point_labels = make_point_from_mask(
                mask, chosen_class)

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])

        predictor.set_image(img)
        predict_mask, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
            )

        save_fig = os.path.join(out_dir, f"fig-{index_number}.png")
        
        r.add(img, None, [point_coords, point_labels], 'Raw image')
        r.add(
            img,
            (mask == chosen_class).astype(np.float16), 
            [point_coords, point_labels], 
            'GT'
        )
        r.add_multiple([{
                'img': img,
                'mask': predict_mask[idx],
                'points': None,
                'title': f'Predict mask {idx}'
            } for idx in range(predict_mask.shape[0])])
        
        r.show_all(save_path=save_fig)
        pass


if __name__ == "__main__":
    point_experiment()
    pass