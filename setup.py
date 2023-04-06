# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

import io
import os
import re


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "segment_anything", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_ALL_REQUIREMENTS = ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"]

_DEV_REQUIREMENTS = [
    "black==23.*",
    "isort==5.12.0",
    "flake8",
    "mypy",
]

extras = {
    "all": _ALL_REQUIREMENTS,
    "dev": _DEV_REQUIREMENTS,
}

setup(
    name="segment_anything",
    license="Apache-2.0",
    author="facebook",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/segment-anything",
    version=get_version(),
    install_requires=get_requirements(),
    packages=find_packages(exclude="notebooks"),
    extras_require=extras,
)
