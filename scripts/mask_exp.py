import torch
import os
from tqdm import tqdm
from scripts.render import Renderer
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from utils import \
    make_nested_dir, get_data_paths, GROUP2, load_file_npz, load_img, resize



def load_model() -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor

def mask_experiment(scale_factor):
    chosen_class = 1
    data_path, mask_path = get_data_paths(GROUP2)
    predictor = load_model()
    mask_input_size = [4*x for x in predictor.model.prompt_encoder.image_embedding_size]
    torch.randn(1, *mask_input_size, dtype=torch.float32)
    prefix = os.path.dirname(data_path[0])
    
    out_dir = os.path.join(
        f"{prefix}-video-segment-with-gt")
    try:
        next_num = max([int(os.path.basename(name)) for name in os.listdir(out_dir)])
        next_num = next_num + 1
    except Exception:
        next_num = 0
        pass

    out_dir = os.path.join(out_dir, f"{next_num}")
    make_nested_dir(out_dir)

    point, label = np.array([[400, 300]]), np.array([1.])
    predict_masks = None

    r = Renderer()
    _mask = None

    for path, mask_path in tqdm(zip(data_path, mask_path),
                                desc="Prediction...",
                                total=len(data_path)):
        img = load_img(path)
        mask = load_file_npz(mask_path)
        # _mask = None

        index_number = int(
            os.path.basename(path).replace(".jpg", "").split('_')[-1])
        
        predictor.set_image(img)

        _mask = resize(mask == chosen_class) * scale_factor
        # _mask = resize(predict_masks[2], [256, 256]) \
        #     if predict_masks is not None else None
        
        # if _mask is not None:
        #     print(_mask.shape)
        #     pass
 
        predict_masks, _, _ = predictor.predict(
            # point_coords=point,
            # point_labels=label,
            multimask_output=True,
            mask_input=_mask,
            return_logits=False,
        )

        save_fig = os.path.join(out_dir, f"fig-{index_number}.png")

        r.add(img, None, [point, label], 'Raw image')
        r.add(
            img,
            (mask == chosen_class).astype(np.float16),
            [point, label], 
            'GT'
        )
        r.add_multiple([{
                'img': None,
                'mask': predict_masks[idx],
                'points': None,
                'title': f'Predict mask {idx}'
            } for idx in range(predict_masks.shape[0])]
            )
        
        r.add(None, np.array(_mask[0]), None, f'Input mask - scale - {scale_factor}')
        
        r.show_all(save_path=save_fig)
        r.reset()

        point = None
        label = None



if __name__ == "__main__":
    for scale_factor in [1.0]:
        print(f"Start exp - {scale_factor}")
        try:
            mask_experiment(scale_factor=scale_factor)
        except Exception as msg:
            print(f"Exp {scale_factor} with excep: {msg}")
    pass