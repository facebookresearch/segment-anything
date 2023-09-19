import torch.nn as nn
from .losses import CustormLoss

AVAI_LOSS = {'ce': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
             'test_custom': CustormLoss, 'mse': nn.MSELoss}


def get_losses(losses):
    loss_dict = {}
    for name in losses:
        assert name in AVAI_LOSS, print('{name} is not supported, please implement it first.'.format(name=name))
        if losses[name].params is not None:
            loss_dict[name] = AVAI_LOSS[name](**losses[name].params)
        else:
            loss_dict[name] = AVAI_LOSS[name]()
    return loss_dict
