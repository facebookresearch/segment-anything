# copyright ziqi-jin
import torch
import torch.nn as nn
from .segment_anything_ori import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter


class BaseExtendSam(nn.Module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, model_type='vit_b'):
        super(BaseExtendSam, self).__init__()
        assert model_type in ['default', 'vit_b', 'vit_l', 'vit_h'], print(
            "Wrong model_type, SAM only can be built as vit_b, vot_l, vit_h and default ")
        self.ori_sam = sam_model_registry[model_type](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(self.ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(self.ori_sam, fix=fix_prompt_en)
        self.mask_adapter = BaseMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de)

    def forward(self, img):
        x = self.img_adapter(img)
        points = None
        boxes = None
        masks = None

        sparse_embeddings, dense_embeddings = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_adapter(
            image_embeddings=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


class SemanticSam(BaseExtendSam):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_prompt_en=False, fix_mask_de=False, class_num=20, model_type='vit_b'):
        super().__init__(ckpt_path=ckpt_path, fix_img_en=fix_img_en, fix_prompt_en=fix_prompt_en,
                         fix_mask_de=fix_mask_de, model_type=model_type)
        self.mask_adapter = SemMaskDecoderAdapter(self.ori_sam, fix=fix_mask_de, class_num=class_num)
