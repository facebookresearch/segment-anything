import torch
from segment_anything.build_sam import build_sam
import os

def extract_prompt_encoding(checkpoint="./sam_vit_b_01ec64.pth", checkpoint_type='vit_b'):
    sam = build_sam[checkpoint_type](checkpoint=checkpoint)
    prompt_encoder = sam.prompt_encoder
    torch.save(
        prompt_encoder.state_dict(), 
        f"{os.path.splitext(checkpoint)[0]}-prompt.pt"
        )
    return prompt_encoder

def convert_to_onnx():
    pass

if __name__ == "__main__":
    prompt_encoder = extract_prompt_encoding()
    pass

