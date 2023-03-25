import torch
import sys
sys.path.append("..")
from sam import build_sam
from predictor import SamPredictor
import numpy as np

from segment_everything import InteractivePredictor
from detectron2.config import LazyConfig
from detectron2.data.detection_utils import read_image

def main():
    test_image_path = "/private/home/mintun/DalleBunny.png"
    point_coords = np.array([[[700,100]]])
    point_labels = np.array([[1]])
    box_coords = np.array([[[500,400], [600, 700]]])
    box_labels = np.array([[2,3]])
    im = read_image(test_image_path, "RGB")

    sam = build_sam()
    sam.eval()
    sam.load_state_dict(torch.load('test_sam6.pth', map_location='cpu'))
    predictor = SamPredictor(sam)
    # Make test_sam6.pth using the convert script.


    cfg = LazyConfig.load("/checkpoint/segment_everything/models/95701_config.yaml")
    checkpoint = "/checkpoint/segment_everything/models/95701_model_final.pth"
    old_model = InteractivePredictor.from_cfg(cfg, checkpoint, device='cpu')


    predictor.set_image(im)
    old_model.set_image(im)
    assert predictor.features.shape == old_model.features['low_res_embeddings'].shape, \
        f"Backbone feature shapes don't match. ({predictor.features.shape} vs. {old_model.features['low_res_embeddings'].shape}"
    assert torch.all(predictor.features == old_model.features['low_res_embeddings']), \
        "Backbone features don't match."
    print("Backbone features match.")


    print("One point run:")
    masks, iou_preds, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=None,
        multimask=True,
    )
    old_model_masks, old_model_iou_preds, old_model_for_next_iter = old_model.predict(
            point_coords=point_coords[0],
            point_labels=point_labels[0],
            from_prev_iter=None,
            return_best_mask_only=False,
            fix_point_eval_of_box_model=True,
            threshold=True,
            is_first_iter=True,
    )
    old_model_low_res_masks = old_model_for_next_iter["low_res_pred_masks"]
    assert masks[0].shape == old_model_masks.shape, \
        f"Mask shapes don't match. ({masks[0].shape} vs. {old_model_masks.shape}"
    assert np.all(masks[0] == old_model_masks), \
        "Masks don't match."
    assert iou_preds[0].shape == old_model_iou_preds.shape, \
        "IoU prediction shapes don't match."
    assert np.all(iou_preds[0] == old_model_iou_preds), \
        "IoU predictions don't match."
    assert low_res_masks.shape == old_model_low_res_masks.shape, \
        "Low res mask shapes don't match."
    assert np.all(low_res_masks == old_model_low_res_masks.numpy()), \
        "Low res masks don't match."
    print("All checks passed.")

    print("Mask refinement run:")
    masks, iou_preds, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=low_res_masks[:,:1,:,:],
        multimask=False,
    )
    old_model_masks, old_model_iou_preds, old_model_for_next_iter = old_model.predict(
            point_coords=point_coords[0],
            point_labels=point_labels[0],
            from_prev_iter={"low_res_pred_masks": old_model_low_res_masks[:,:1,:,:]},
            return_best_mask_only=False,
            fix_point_eval_of_box_model=True,
            threshold=True,
            is_first_iter=False,
    )
    old_model_low_res_masks = old_model_for_next_iter["low_res_pred_masks"]
    assert masks[0].shape == old_model_masks.shape, \
        "Mask shapes don't match."
    assert np.all(masks[0] == old_model_masks), \
        "Masks don't match."
    assert iou_preds[0].shape == old_model_iou_preds.shape, \
        "IoU prediction shapes don't match."
    assert np.all(iou_preds[0] == old_model_iou_preds), \
        "IoU predictions don't match."
    assert low_res_masks.shape == old_model_low_res_masks.shape, \
        "Low res mask shapes don't match."
    assert np.all(low_res_masks == old_model_low_res_masks.numpy()), \
        "Low res masks don't match."
    print("All checks passed.")

    print("Box test:")
    masks, iou_preds, low_res_masks = predictor.predict(
        point_coords=box_coords,
        point_labels=box_labels,
        mask_input=None,
        multimask=True,
    )
    old_model_masks, old_model_iou_preds, old_model_for_next_iter = old_model.predict(
            point_coords=box_coords[0],
            point_labels=box_labels[0],
            from_prev_iter=None,
            return_best_mask_only=False,
            fix_point_eval_of_box_model=True,
            threshold=True,
            is_first_iter=True,
    )
    old_model_low_res_masks = old_model_for_next_iter["low_res_pred_masks"]
    assert masks[0].shape == old_model_masks.shape, \
        "Mask shapes don't match."
    assert np.all(masks[0] == old_model_masks), \
        "Masks don't match."
    assert iou_preds[0].shape == old_model_iou_preds.shape, \
        "IoU prediction shapes don't match."
    assert np.all(iou_preds[0] == old_model_iou_preds), \
        "IoU predictions don't match."
    assert low_res_masks.shape == old_model_low_res_masks.shape, \
        "Low res mask shapes don't match."
    assert np.all(low_res_masks == old_model_low_res_masks.numpy()), \
        "Low res masks don't match."
    print("All checks passed.")


if __name__=="__main__":
    main()
