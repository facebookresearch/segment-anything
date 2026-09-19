<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 最新动态 -- SAM 2: 图像与视频中的分割一切

欢迎体验全新发布的 [**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/segment-anything-2)。

* SAM 2 代码仓库：https://github.com/facebookresearch/segment-anything-2
* SAM 2 在线演示：https://sam2.metademolab.com/
* SAM 2 论文：https://arxiv.org/abs/2408.00714

 ![SAM 2 架构图](https://github.com/facebookresearch/segment-anything-2/blob/main/assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** 是用于解决图像与视频中基于提示（Promptable）视觉分割的基础模型。我们将图像视为单帧视频，从而将 SAM 扩展到了视频领域。模型设计采用精简的 Transformer 架构并配备流式记忆机制，以实现实时视频处理。我们构建了一个模型在环（Model-in-the-loop）数据引擎，通过用户交互协同提升模型与数据质量，并据此构建了迄今为止规模最大的视频分割数据集 [**SA-V 数据集**](https://ai.meta.com/datasets/segment-anything-video)。在我们的数据上完成训练的 SAM 2 在广泛的任务与视觉领域中均表现出优异性能。

# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`论文`](https://ai.facebook.com/research/publications/segment-anything/)] [[`项目主页`](https://segment-anything.com/)] [[`在线演示`](https://segment-anything.com/demo)] [[`数据集`](https://segment-anything.com/dataset/index.html)] [[`博客`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX 引用`](#citing-segment-anything)]

![SAM 设计架构](assets/model_diagram.png?raw=true)

**Segment Anything Model (SAM)** 可以根据点（points）或边界框（boxes）等输入提示生成高质量的对象掩码（masks），并且能够用于为图像中的所有对象生成掩码。该模型在包含 1100 万张图像与 11 亿个掩码的[数据集](https://segment-anything.com/dataset/index.html)上进行了训练，并在多种分割任务上展现出强大的零样本（Zero-Shot）性能。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## 安装

运行代码需要 `python>=3.8`，以及 `pytorch>=1.7` 和 `torchvision>=0.8`。请遵循[此处](https://pytorch.org/get-started/locally/)的指引安装 PyTorch 与 TorchVision 依赖。强烈建议安装支持 CUDA 的 PyTorch 与 TorchVision。

安装 Segment Anything：

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

或将本仓库克隆至本地后执行安装：

```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

以下可选依赖项用于掩码后处理、以 COCO 格式保存掩码、运行示例 Notebook 以及将模型导出为 ONNX 格式。运行示例 Notebook 还需要安装 `jupyter`。

```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="GettingStarted"></a>快速入门

首先下载一个[模型权重 (Checkpoint)](#model-checkpoints)。随后只需数行代码，即可根据给定提示生成掩码：

```python
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

或为整张图像自动生成全部掩码：

```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

此外，也可以直接通过命令行对图像生成掩码：

```bash
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

详细信息请参阅示例 Notebook：[基于提示使用 SAM](/notebooks/predictor_example.ipynb) 以及[自动生成掩码](/notebooks/automatic_mask_generator_example.ipynb)。

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX 导出

SAM 拥有轻量级的掩码解码器（Mask Decoder），可以导出为 ONNX 格式，从而在任何支持 ONNX Runtime 的环境中运行，例如在[在线演示](https://segment-anything.com/demo)中所展示的浏览器端直接运行。执行以下命令导出模型：

```bash
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

有关如何将 SAM 骨干网络（Backbone）的图像预处理与 ONNX 模型的掩码预测相结合的详细信息，请参阅[示例 Notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)。建议使用最新稳定版本的 PyTorch 进行 ONNX 导出。

### Web 前端演示

`demo/` 目录提供了一个简洁的单页 React 应用程序，展示了如何在 Web 浏览器中通过多线程利用导出的 ONNX 模型运行掩码预测。详情请参阅 [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md)。

## <a name="Models"></a>模型权重 (Model Checkpoints)

提供了三种具有不同骨干网络（Backbone）尺寸的模型版本。可通过以下代码实例化模型：

```python
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

点击下方链接下载对应模型类型的权重文件：

- **`default` 或 `vit_h`: [ViT-H SAM 模型权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM 模型权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM 模型权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## 数据集

数据集概览请见[此处](https://ai.facebook.com/datasets/segment-anything/)。数据集可从[此处下载](https://ai.facebook.com/datasets/segment-anything-downloads/)。下载数据集即表示您已阅读并同意 SA-1B 数据集研究许可协议（SA-1B Dataset Research License）。

我们为每张图像保存一个包含掩码信息的 JSON 文件。在 Python 中可以按如下格式加载为字典：

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # 图像 ID
    "width"                 : int,              # 图像宽度
    "height"                : int,              # 图像高度
    "file_name"             : str,              # 图像文件名
}

annotation {
    "id"                    : int,              # 标注 ID
    "segmentation"          : dict,             # 以 COCO RLE 格式保存的掩码
    "bbox"                  : [x, y, w, h],     # 掩码的外接边界框，XYWH 格式
    "area"                  : int,              # 掩码的像素面积
    "predicted_iou"         : float,            # 模型自身对掩码质量的预测 IoU
    "stability_score"       : float,            # 掩码质量的稳定性度量指标
    "crop_box"              : [x, y, w, h],     # 用于生成掩码的图像裁剪区域，XYWH 格式
    "point_coords"          : [[x, y]],         # 输入给模型以生成掩码的点坐标
}
```

图像 ID 可以在 `sa_images_ids.txt` 中找到，同样可以通过上述[链接](https://ai.facebook.com/datasets/segment-anything-downloads/)下载。

要将 COCO RLE 格式的掩码解码为二值矩阵（Binary Mask）：

```python
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

更多操作 RLE 格式掩码的说明请参阅[此处](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)。

## 开源协议

本模型遵循 [Apache 2.0 许可证](LICENSE)。

## 贡献指南

请参阅[贡献指南](CONTRIBUTING.md)以及[行为准则](CODE_OF_CONDUCT.md)。

## 贡献者

Segment Anything 项目的顺利实现离不开众多贡献者的付出（按首字母排序）：

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## 引用 Segment Anything

如果您在研究中使用了 SAM 或 SA-1B，请使用以下 BibTeX 条目进行引用：

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月19日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
