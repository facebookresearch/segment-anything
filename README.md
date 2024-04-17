# Segment Anything (SAM) for Nuke

## Introduction

This project brings Meta's powerful **Segment Anything Model (SAM)** to **The Foundry's Nuke**. **Segment Anything** is a state-of-the-art neural network for creating precise masks around objects in single images, capable of handling both familiar and unfamiliar subjects without additional training.

This project offers a native integration within Nuke, requiring no external dependencies or complex installation. The neural network is wrapped into an intuitive **Gizmo**, controllable via Nuke's standard Tracker for a seamless experience.

With this implementation, you gain access to cutting-edge object segmentation capabilities directly inside your Nuke workflow, leveraging **Segment Anything** to isolate and extract objects in time efficinet manner.  streamlining your compositing tasks.

This project implements Meta's [**Segment Anything Model (SAM)** - https://segment-anything.com/](https://segment-anything.com/) for **The Foundry's Nuke**.

**Segment Anything** is a **mask creation neural network** for single images, capable of segmenting familiar and unfamiliar objects without addition trainig.

This implementation allows **Segment Anything** to be used **natively** inside Nuke without any external dependencies or complex installations. It wraps the network in an **easy-to-use Gizmo**, controllable with familiar native's Nuk's Tracker.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>


## Features

- **Intuitive interface** for selecting objects using Nuke's familiar Tracker node.
- **Three levels of quality**, allowing users to balance precision and GPU memory usage.
- **Preprocessing stage** with an encoded matte for reusing and speeding up multiple object selections.
- **Efficient memory usage** - the high-quality model fits on most 8GB graphics cards, while the low-quality model is compatible with 4GB cards.
- **Nuke 13 compatibility**. Note: **Preprocessing is recommended** for an optimal experience.

## Compatibility

**Nuke 13.2+**, tested on **Linux** and **Windows**.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/Segment-Anything-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**Segment Anything** will then be accessible under the toolbar at **Cattery > Segmentation > SegmentAnything**.

### ⚠️ Extra Steps for Nuke 13

4. Add the path for **RIFE** to your `init.py`:
``` py
import nuke
nuke.pluginAddPath('./Cattery/SegmentAnything')
```

5. Add an menu item to the toolbar in your `menu.py`:

``` py
import nuke
toolbar = nuke.menu("Nodes")
toolbar.addCommand('Cattery/Optical Flow/SegmentAnything', 'nuke.createNode("SegmentAnything")', icon="SAM.png")
```

## License and Acknowledgments

**SegmentAnything.cat** is licensed under the MIT License, and is derived from https://github.com/facebookresearch/segment-anything.

While the MIT License permits commercial use of **ViTMatte**, the dataset used for its training may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/facebookresearch/segment-anything for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of RIFE.cat.**

## Citation

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
