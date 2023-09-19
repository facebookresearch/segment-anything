'''
@copyright ziqi-jin
You can create custom loss function in this file, then import the created loss in ./__init__.py and add the loss into AVAI_LOSS
'''
import torch.nn as nn


# example
class CustormLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, y):
        pass