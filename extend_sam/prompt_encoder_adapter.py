# copyright ziqi-jin

import torch.nn as nn
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params


class BasePromptEncodeAdapter(nn.Module):

    def __init__(self, ori_sam: Sam, fix=False):
        super(BasePromptEncodeAdapter, self).__init__()

        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            fix_params(self.sam_prompt_encoder)

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings
