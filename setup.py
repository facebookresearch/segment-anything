from setuptools import setup, find_packages

packages = find_packages()
print("packages: ", packages)
setup(
    name='pytorch-labs-segment-anything-fast',
    version='0.2',
    packages=packages,
    install_requires=[
        'torch>=2.2.0.dev20231026',
        'torchvision>=0.17.0.dev20231026',
        'diskcache',
        'pycocotools',
        'scipy',
        'scikit-image',
        'torchao',
    ],
    include_package_data=True,
    package_data={
        'segment_anything_fast.configs': ["*.p"],
    },
    description='A pruned, quantized, compiled, nested and batched implementation of segment-anything',
    long_description_content_type='text/markdown',
    url='https://github.com/pytorch-labs/segment-anything-fast',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
