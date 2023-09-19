import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, VisionDataset
import numpy as np


class BaseSemanticDataset(VisionDataset):
    """
    if you want to customize a new dataset to train the segmentation task,
    the img and mask file need be arranged as this sturcture.
        ├── data
        │   ├── my_dataset
        │   │   ├── img
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann
        │   │   │   ├── train
        │   │   │   │   ├── xxx{ann_suffix}
        │   │   │   │   ├── yyy{ann_suffix}
        │   │   │   │   ├── zzz{ann_suffix}
        │   │   │   ├── val
    """

    def __init__(self, metainfo, dataset_dir, transform, target_transform,
                 image_set='train',
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 data_prefix: dict = dict(img_path='img', ann_path='ann'),
                 return_dict=False):
        '''

        :param metainfo: meta data in original dataset, e.g. class_names
        :param dataset_dir: the path of your dataset, e.g. data/my_dataset/ by the stucture tree above
        :param image_set: 'train' or 'val'
        :param img_suffix: your image suffix
        :param ann_suffix: your annotation suffix
        :param data_prefix: data folder name, as the tree shows above, the data_prefix of my_dataset: img_path='img' , ann_path='ann'
        :param return_dict: return dict() or tuple(img, ann)
        '''
        super(BaseSemanticDataset, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = metainfo['class_names']
        self.img_path = os.path.join(dataset_dir, data_prefix['img_path'], image_set)
        self.ann_path = os.path.join(dataset_dir, data_prefix['ann_path'], image_set)
        print('img_folder_name: {img_folder_name}, ann_folder_name: {ann_folder_name}'.format(
            img_folder_name=self.img_path, ann_folder_name=self.ann_path))
        self.img_names = [img_name.split(img_suffix)[0] for img_name in os.listdir(self.img_path) if
                          img_name.endswith(img_suffix)]
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index] + self.img_suffix))
        ann = Image.open(os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)

        if self.return_dict:
            data = dict(img_name=self.img_names[index], img=img, ann=ann,
                        img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
                        ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
            return data
        return img, ann

    def __len__(self):
        return len(self.img_names)


class VOCSemanticDataset(Dataset):
    def __init__(self, root_dir, domain, transform, with_id=False, with_mask=False):
        super(VOCSemanticDataset, self).__init__()
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'

        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt' % domain).readlines()]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class TorchVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super(TorchVOCSegmentation, self).__init__(root=root, year=year, image_set=image_set, download=download,
                                                   transform=transform, target_transform=target_transform)
        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = np.array(target)
        return img, target
