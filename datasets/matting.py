import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import numpy as np

class BaseMattingDataset(VisionDataset):
    """
    if you want to customize a new dataset to train the matting task,
    the img and mask file need be arranged as this sturcture.
        ├── data
        │   ├── my_dataset
        │   │   ├── img
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── trimap
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
                 trimap_transform=None,
                 image_set='train',
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 trimap_suffix=None,
                 data_prefix: dict = dict(img_path='img', ann_path='ann', trimap_path='trimap_pth'),
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
        super(BaseMattingDataset, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = metainfo['class_names']
        self.img_path = os.path.join(dataset_dir, data_prefix['img_path'], image_set)
        self.ann_path = os.path.join(dataset_dir, data_prefix['ann_path'], image_set)

        print('img_folder_name: {img_folder_name}, ann_folder_name: {ann_folder_name}'.format(
            img_folder_name=self.img_path, ann_folder_name=self.ann_path))
        self.img_names = [img_name.split(img_suffix)[0] for img_name in os.listdir(self.img_path) if
                          img_name.endswith(img_suffix)]
        
        self.has_trimap = trimap_suffix is not None
        if self.has_trimap:
            self.trimap_path = os.path.join(dataset_dir, data_prefix['trimap_pth'], image_set)
            print('trimap_folder_name: {trimap_folder_name}'.format(trimap_folder_name=self.trimap_path))
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict
        self.trimap_transform = trimap_transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index] + self.img_suffix))
        ann = Image.open(os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)
        if self.has_trimap:
            ## return for self.has_trimpa==True
            trimap = Image.open(os.path.join(self.trimap_path, self.img_names[index] + self.trimap_suffix))
            if self.trimap_transform:
                trimap = self.trimap_transform(trimap)
            else:
                print("Warnning: you may need set transform function for trimap input")
            if self.return_dict:
                data = dict(img_name=self.img_names[index], img=img, ann=ann, trimap=trimap,
                            img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
                            ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix),
                            trimap_path=os.path.join(self.trimap_path, self.img_names[index] + self.trimap_suffix))
                return data
            return img, ann, trimap
        else:
            ## return for self.has_trimpa==False
            if self.return_dict:
                data = dict(img_name=self.img_names[index], img=img, ann=ann,
                            img_path=os.path.join(self.img_path, self.img_names[index] + self.img_suffix),
                            ann_path=os.path.join(self.ann_path, self.img_names[index] + self.ann_suffix))
                return data
            return img, ann

    def __len__(self):
        return len(self.img_names)

