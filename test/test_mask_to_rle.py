import torch
import itertools
from segment_anything_fast.utils.amg import (
    mask_to_rle_pytorch,
    mask_to_rle_pytorch_2,
)

def test_masks(masks):
    rles_0 = mask_to_rle_pytorch(masks)
    rles_2 = mask_to_rle_pytorch_2(masks)
    
    for i in range(len(rles_0)):
        torch.testing.assert_close(torch.tensor(rles_0[i]['counts']), torch.tensor(rles_2[i]['counts']))

for b, w, h in itertools.product([1, 5], [50, 128], [50, 128]):
    test_masks(torch.randn(b, w, h).clamp(min=0).bool().cuda())
    test_masks(torch.randn(b, w, h).mul(0).bool().cuda())
    test_masks(torch.randn(b, w, h).mul(0).add(1).bool().cuda())
