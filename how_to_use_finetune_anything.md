# How to use finetune-anything
finetune-anything (FA) is intended as a tool to help users quickly build extended SAM models. It not only supports the built-in basic tasks and basic models, but also supports user-defined extensions of different modules, training processes, and datasets for the extend SAM.

- Content
    - [Structure](#Structure)
    - [Model](#Model)
    - [Datasets](#Datasets)
    - [Losses](#Losses)
    - [Optimizer](#Optimizer)
    - [Runner](#Runner)
    - [Logger](#Logger)
    - [One more thing](#One-more-thing)


## Structure
Using FA can be divided into two parts: training and testing. The training part includes [model](#Model), [Datasets](#Datasets), [Losses](#Losses), [Optimizer](#Optimizer), [Logger](#Logger), and [Runner](#Runner).
The above content needs to be configured through the yaml file in `config`. 
- The tasks already supported by FA can be trained and tested directly by inputting `task_name`.
```
CUDA_VISIBLE_DEVICES=${your GPU number} python train.py --task_name ${one of supported task names}
```
- Custom configuration files can be trained and tested by reading `cfg`
```
CUDA_VISIBLE_DEVICES=${your GPU number} python train.py --cfg config/${yaml file name}
```
The testing part is coming soon ~

## Model
The SAM model includes image encdoer, prompt encoder and mask decoder. FA further encapsulates the encoder and decoder of SAM and identify Extend-SAM model consists of image encoder adapter, prompt encoder adapter and mask decoder adapter. The initialized process of Extend-SAM as below,
<img width="960" src="https://user-images.githubusercontent.com/67993288/248108534-62a4e5aa-cf4f-41f9-b745-db2924a376bc.svg">

Users can choose the adapter that need to be fixed or learned during the finetune process. This function can be configured in the `model` part of the yaml file, as shown in the following example:

```yaml
model:
sam_name: 'extend sam name' # e.g., 'sem_sam', custom SAM model name, you should implement this model('sem_sam') first
params:
  # Fix the a part of parameters in SAM
  fix_img_en: True  # fix image encoder adapter parameters
  fix_prompt_en: True # fix prompt encoder adapter parameters
  fix_mask_de: False # unfix mask decoder adapter parameters to learn
  ckpt_path: 'your original sam weights'  # e.g., 'sam_ckpt/sam_vit_b_01ec64.pth' 
  class_num: 21 # number of classes for your dataset(20) + background(1)
  model_type: 'vit_b'    # type should be in [vit_h, vit_b, vit_l, default], this is original SAM type 
                         # related to different original SAM model. the type should be corresponded to the ckpt_path
```
### Customized Model
If you need to redesign the structure of a certain module of SAM, you need to write code according to the following three steps. Take [SemanticSAM](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/extend_sam.py#L43) as an example.
- step1

First, inherit the corresponding adapter base class in `extend_sam\xxx_(encoder or decoder)_adapter.py`, and then implement the `__init__` and `forward` function corresponding to the adapter.
```python
class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False, class_num=20):
        super(SemMaskDecoderAdapter, self).__init__(ori_sam, fix) # init super class
        self.decoder_neck = MaskDecoderNeck(...) # custom module
        self.decoder_head = SemSegHead(...) # custom module
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck) # give the weights which are with the same name in original SAM to customized module
        self.pair_params(self.decoder_head)

    def forward(self, ...):
        ... = self.decoder_neck(...)
        masks, iou_pred = self.decoder_head(...)
        return masks, iou_pred
```
- step2

First inherit the BaseExtendSAM base class in [extend_sam.py](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/extend_sam.py#L43), and make necessary modifications to `__init__` function.
```python
class SemanticSam(BaseExtendSam):

    def __init__(self, ...):
        super().__init__(...) # init super class
        self.mask_adapter = SemMaskDecoderAdapter(...) # replace original Adapter as the  new identified customized Adapter 
```
- step3

Add new Extend-SAM class to [AVAI_MODEL](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/__init__.py#L10) dict and give it a key.
then you can train this new model by modify the `sam_name` in config file.

## Datasets

FA comes with datasets for multiple tasks, and also supports custom datasets, and sets the training and test datasets separately. Takes `torch_voc_sem` as an example, the configuration file of the dataset part is as follows,
The dataset part includes `name`, `params`, `transforms` and `target_transforms`,
The `params` which is a `dict` include the key and value your want to set about the init function's parameters of corresponding dataset. make sure the dataset has parameters with the same names as the key.
`transforms` and `target_transforms` respectively correspond to the input image and Ground Truth for transform processing.
`transforms/target_transforms` support to set the implemented transform function and the corresponding `params`, `params` are still in the form of a `dict`, and transform will process the datasets according to the input order of the configuration file.
```yaml
  # Dataset
  dataset:
    name: 'torch_voc_sem'
    params:
      root: '/your/dataset/path/'
      year: '2012'
      image_set: 'train'
    transforms:
      resize:
        params:
          size: [1024, 1024]
      to_tensor:
        params: ~ # no parameters, set to '~'
    target_transforms:
      resize:
        params:
          size: [1024, 1024]
```

### Customized Dataset

### Customized Transform

If you want to customize the transform, you can follow the following three steps,

- step1

    - Torch-supported transform, skip this step.
    
    - Torch-unsupported transform
    
    Create it in [datasets/transforms.py](https://github.com/ziqi-jin/finetune-anything/blob/main/datasets/transforms.py),  implement the `__init__` and `forward` function.
    
```python
import torch.nn as nn
class CustomTransform(nn.Module):
    def __init__(self):
    # identify your init process here
    def forward(self):
    # identify your transform process here
```
    
    
- step2

    Import torch-supported transform you want or torch-unsupported transform your identify in [datasets/transforms.py](https://github.com/ziqi-jin/finetune-anything/blob/main/datasets/transforms.py).
    Then add this transform into the AVIAL_TRANSFORM  dict, give this transform a key like `resize`, and the value is the transform class.
    
```python
import torchvision.transforms as T
AVIAL_TRANSFORM = {'your_transform_name': T.XXX, 'your_transform_name': CustomTransform}
```
  
- step3
    
    Set the loss in your config file.
```yaml
    transforms:
      your_transform_name:
        params: # if there are parameters of the transform's __init__ function to be set. else set to '~'
          params_1: xxx
          params_2: xxx
```
    
## Losses

FA supports multiple torch loss functions, and also allows users to customize the loss function. The configuration content of the loss function part is as below,
```yaml
losses:
    ce:
      weight: 0.5
      params:  # the initial params of loss could be identified here
        ignore_index: 255
      label_one_hot: False
    mse:
      weight: 5.0
      params: ~ # no parameters, set '~'
      label_one_hot: True
```
Now loss part has `weight`, `params`, and `label_one_hot` keys, `weight` control the weight of each loss in total loss. Take the config above as example, assume the `ce` loss as $Loss_{ce}$ and the `mse` as $Loss_{mse}$, the final total loss as below,

$$
Loss_{total} = weight_{ce} \times Loss_{ce} + weight_{mse} \times Loss_{mse} = 0.5 \times Loss_{ce} + 5.0 \times Loss_{mse}
$$

The `params` which is a `dict` include the key and value your want to set about the corresponding loss function's parameters, make sure the loss function has parameters with the same names as the key. if you don't need the set params, give params `~`.
for semantic segmentation task, if your loss function need a one hot label, set the `label_one_hot` to `True`.


### Customized Losses

If you want to customize the loss function, you can follow the following three steps,

- step1

    - Torch-supported Loss, skip this step.
    
    - Torch-unsupported Loss
    
    Create it in [loss.py](https://github.com/ziqi-jin/finetune-anything/blob/main/losses/losses.py),  implement the `__init__` and `forward` function.
    
```python
import torch.nn as nn
class CustormLoss(nn.Module):
def __init__(self,xxx):
    # identify your init process here
def forward(self, x, y, xxx):
    # identify your forward process here
```
    
    
- step2

    Import torch-supported loss you want or torch-unsupported loss your identify in [losses/\_\_init\_\_,py](https://github.com/ziqi-jin/finetune-anything/blob/26b9ebd1b035a2f0ec8ce4e358eac79de7e263a2/losses/__init__.py#L2).
    Then add this loss into the AVAI_LOSS dict, give this loss a key like `ce`, and the value is the loss function.
    
```python

import torch.nn as nn
from .losses import YourCuntomLoss
AVAI_LOSS = {'your loss key': YourCuntomLoss, 'your loss key': nn.xxxLoss}
```
  
- step3
    
    Set the loss in your config file.
    
```yaml
losses:
    your_loss_key:
      weight: your_weight # float
      params:  
        your_loss_param1: xx
        your_loss_param2: xx
      label_one_hot: False
```

## Optimizer
FA's optimizer supports setting learning_rate(`lr`) and weight_decay(`wd`) for any module in the adapter that is not fixed.
User could use keyword `sgd`, `adam`, and `adamw` to set the optimizer. the `opt_params` save necessary params for each kind of optimizer.
- Normal module setting

`lr_default` save the default learing rate for all unfixed params, `wd_default` save the default weight decay for all unfixed params, 
`momentum` save the momentum for optimizer. if the corresponding optimizer has no parameter, e.g., `adam` has no `momentum`, just set the `momentum` to `~`.
- Specific module setting

The left three params `group_keys`, `lr_list` and `wd_list` is for specific module.
They are list have the same length and correspond to the module name, learning rate and weight decay respectively. 
for example, if you want to give `mask_adapter.decoder_head.output_hypernetworks_mlps` module a specific optimizing parameter, put it into `group_keys` as a list first, and then set the corresponding learning rate and weight decay into `lr_list` and `wd_list`.
If there are multiple modules that need to use the same specific parameter setting, just add the key to the corresponding list in the `group_keys`. For example, add `modulexxx` to the first list of `group_keys`.
```yaml
  # Optimizer
  opt_params:
    lr_default:  1e-3
    wd_default: 1e-4
    momentum: 0.9
    group_keys: [ [ 'mask_adapter.decoder_head.output_hypernetworks_mlps', 'modulexxx' ], ['second_module'], ]
    lr_list:  [ 1e-2, 1e-4, ]
    wd_list:  [ 0.0, 0.1, ]
  opt_name: 'sgd' # 'sgd'
  scheduler_name: 'cosine'
```
FA also supports multiple schedulers, which can be set using the keyword `single_step`, `multi_step`, `warmup_multi_step`, `cosine`, `linear`.
## Runner

## Logger
As shown in the config file, FA provides two kinds of loggers, one is the log output by default and will be saved in `log_folder`, and the other is the log output of tensorboard saved in `tensorboard_folder` when `use_tensorboard` is `True`.
The best model will be saved in `model_folder`.
```yaml
  # Logger
  use_tensorboard: True
  tensorboard_folder: './experiment/tensorboard'
  log_folder: './experiment/log'
  model_folder: './experiment/model'
```

## One more thing

If you need to use loss, dataset, or other functions that are not supported by FA, please submit an issue, and I will help you to implement them. At the same time, developers are also welcome to develop new loss, dataset or other new functions for FA, please submit your PR (pull requests).