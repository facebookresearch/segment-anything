import torchvision.transforms as T
from omegaconf.dictconfig import DictConfig
import torch.nn as nn

AVIAL_TRANSFORM = {'resize': T.Resize, 'to_tensor': T.ToTensor}


def get_transforms(transforms: DictConfig):
    T_list = []
    for t_name in transforms.keys():
        assert t_name in AVIAL_TRANSFORM, "{T_name} is not supported transform, please implement it and add it to " \
                                          "AVIAL_TRANSFORM first.".format(T_name=t_name)
        if transforms[t_name].params is not None:
            T_list.append(AVIAL_TRANSFORM[t_name](**transforms[t_name].params))
        else:
            T_list.append(AVIAL_TRANSFORM[t_name]())
    return T.Compose(T_list)


class CustomTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
