import torch
import pandas as pd

def create_result_entry(anns, gt_masks_list, masks, scores, img_idx):
    argmax_scores = torch.argmax(scores, dim=1)
    inference_masks = masks.gather(1, argmax_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        (masks.size(0), 1, masks.size(2), masks.size(3)))).squeeze(1)

    def _iou(mask1, mask2):
        assert mask1.dim() == 3
        assert mask2.dim() == 3
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        return (intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2)))

    top_score_ious = _iou(inference_masks, gt_masks_list)

    entry = []
    for idx in range(top_score_ious.size(0)):
        entry.append(
            [img_idx, anns[idx]['id'], anns[idx]['category_id'], top_score_ious[idx]])
    return entry


def calculate_miou(results, mask_debug_out_dir, silent, cat_id_to_cat):
    df = pd.DataFrame(results, columns=['img_id', 'ann_id', 'cat_id', 'iou'])
    df.to_csv(f'{mask_debug_out_dir}/df.csv')
    df['supercategory'] = df['cat_id'].map(
        lambda cat_id: cat_id_to_cat[cat_id]['supercategory'])
    df['category'] = df['cat_id'].map(
        lambda cat_id: cat_id_to_cat[cat_id]['name'])

    # TODO: cross reference the specifics of how we calculate mIoU with
    # the SAM folks (should it be per dataset, per category, per image, etc)
    # currently, just calculate them all

    # TODO: QOL save the summaries to file

    # per category
    per_category = pd.pivot_table(
        df, values='iou', index=['cat_id', 'supercategory', 'category'],
        aggfunc=('mean', 'count'))
    if not silent:
        print('\nmIoU averaged per category')
        print(per_category)

    # per super-category
    per_supercategory = pd.pivot_table(
        df, values='iou', index=['supercategory'],
        aggfunc=('mean', 'count'))
    if not silent:
        print('\nmIoU averaged per supercategory')
        print(per_supercategory)

    # per all selected masks
    per_all_masks_agg = df['iou'].agg(['mean', 'count'])
    if not silent:
        print('\nmIoU averaged per all selected masks')
        print(per_all_masks_agg)

    return df['iou'].agg(['mean', 'count'])['mean']
