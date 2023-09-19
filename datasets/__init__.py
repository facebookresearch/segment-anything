from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset, TorchVOCSegmentation
from .transforms import get_transforms
from torchvision.datasets import VOCSegmentation

segment_datasets = {'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
                    'voc_sem': VOCSemanticDataset, 'torch_voc_sem': TorchVOCSegmentation}
det_dataset = {'base_det': BaseDetectionDataset, }


def get_dataset(cfg):
    name = cfg.name
    assert name in segment_datasets or name in det_dataset, \
        print('{name} is not supported, please implement it first.'.format(name=name))
    # TODO customized dataset params:
    # customized dataset params example:
    # if xxx:
    #   param1 = cfg.xxx
    #   param2 = cfg.xxx
    # return name_dict[name](path, model, param1, param2, ...)
    transform = get_transforms(cfg.transforms)
    if name in det_dataset:
        return det_dataset[name](**cfg.params, transform=transform)
    target_transform = get_transforms(cfg.target_transforms)
    return segment_datasets[name](**cfg.params, transform=transform, target_transform=target_transform)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data
