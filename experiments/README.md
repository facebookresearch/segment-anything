To run the experiments you need to update the script paths and install fire, pandas and tqdm

## Model Checkpoints

Need checkpoints from https://github.com/facebookresearch/segment-anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

## COCO2017 dataset

Need to download

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

## Folder structure of experimental data
```
experiments_data/tmp
experiments_data/tmp/sam_coco_mask_center_cache
experiments_data/tmp/sam_eval_masks_out
experiments_data/datasets
experiments_data/datasets/coco2017
experiments_data/datasets/coco2017/val2017
experiments_data/datasets/coco2017/annotations
experiments_data/checkpoints
```
## Environment details

### Hardware
These  experiments were run on an Amazon p4d.24xlarge instance. See the Product details of the EC2 website for the exact details. A few key highlights are

- 8 A100 GPUs with 40960MiB running at 400W
- 96 vCPUs
- 1152 GiB of RAM
- Software


### Versions

- PyTorch nightly and Python 3.10
- https://github.com/cpuhrsch/segment-anything fork of https://github.com/facebookresearch/segment-anything with additional commits if you want to reproduce baseline and first few experiments
- This https://github.com/pytorch-labs/segment-anything-fast

### Installation instructions

```
$ conda create -n nightly20231117py310
$ conda activate nightly20231117py310
$ conda install python=3.10
$ pip install https://download.pytorch.org/whl/nightly/cu121/torch-2.2.0.dev20231117%2Bcu121-cp310-cp310-linux_x86_64.whl
$ pip install https://download.pytorch.org/whl/nightly/cu121/torchvision-0.17.0.dev20231117%2Bcu121-cp310-cp310-linux_x86_64.whl
$ git clone https://github.com/cpuhrsch/segment-anything.git
$ cd segment-anything
$ pip install -e .
$ cd ..
$ git clone https://github.com/pytorch-labs/segment-anything-fast.git
$ cd segment-anything-fast
$ pip install -e .
```

If you plan to run the scripts that run the experiments from segment-anything-fast it is important to install the segment-anything fork in editable mode so that the script can switch between different commits of the fork automatically.


### How to run experiments

```
$ python run_experiments.py 16 vit_b <pytorch_github> <segment-anything_github> <path_to_experiments_data> --run-experiments --num-workers 32
```

If at any point you run into issue, please note that you can increase verbosity by adding `--capture_output False` to above command. Also, please don't hesitate to open an issue.


### Data
We are using the COCO2017 Validation (Val images) dataset. We use this dataset to serve as a somewhat realistic distribution of input images and aim to measure a) accuracy and b) performance.
Measurement
Accuracy
Our main goal is to verify that our performance optimizations do not degrade the accuracy of the model. We do not aim to reproduce any paper results or aim to make statements about the accuracy of this model on the dataset. This measurement serves as an additional integration test in conjunction with numerous unit and other separate integration tests.

We calculate the center points of the mask annotations using a rudimentary version of https://arxiv.org/pdf/2304.02643.pdf, section D.1.Point Sampling ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/data.py#L10-L120)). These center points serve as annotations per image. Note that the number of masks and thus number of annotations per image vary.

These images and annotations are given to the predict_torch method of an instance of SamPredictor to predict masks. These are then compared to the ground truth masks using the Intersection over Union (IoU) metric ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/metrics.py#L4-L22)). We calculate the mean IoU (mIoU) metric over the entire 5000 images of the validation dataset to track accuracy.
Performance
Our goal is to measure the runtime of PyTorch models. We purposefully exclude data movements or calculation of the metrics. Specifically we measure the execution time on the GPU of running the image encoder (e.g. vit_h) and SamPredictor.predict_torch ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/eval_combo.py#L127-L165), [code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/eval_combo.py#L68-L99)).

Each experiment is run in a separate Python process created from scratch. We run three batches of warmup before each experiment. This also implies that we are excluding compilation time from benchmarking. 

We measure the execution time and calculate the number of images that can be processed per image (img/s). We also measure the maximum amount of memory allocated at the end of the process using torch.cuda.max_memory_allocated.
Tracing

We collect kernel and memory traces using PyTorch native tooling and analyze it with [Perfetto UI](https://perfetto.dev/). When collecting these traces and profiles we typically only limit us to a few batches. Otherwise the files can become very large and difficult to load.

### Kernel traces

One can write a simple wrapper that runs a function under the tracer context and writes out the result to a compressed json file. The resulting chrome trace can then be analyzed with Perfetto UI.

```
def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result
```

It can be very useful to annotate certain regions in these traces to map (pieces of) the code to the overall traces. For this we frequently use record_function. Consider the following as an example.

```
with torch.autograd.profiler.record_function("timed region"):
    with torch.autograd.profiler.record_function("image encoder"):
        features_batch = encoder(input_image_batch)
        features_batch = features_batch[:orig_input_image_batch_size]

    with torch.autograd.profiler.record_function("nt predict_torch"):
        predictor.reset_image()
[...]
```

### Memory profiles

We record the memory history and use memory_viz.py to convert the result into a human readable html file.

```
def memory_runner(path, fn, *args, **kwargs):
    print("Start memory recording")
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(
        True, 
        trace_alloc_max_entries=100000,           
        trace_alloc_record_context=True
    )
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    print("Finish memory recording")
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(snapshot, f)
    # Use to convert pickle file into html
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html
    return result
```
