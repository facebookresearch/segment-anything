import subprocess
import fire
import itertools
import functools

sam_commits = {
    "default": "6fdee8f2727f4506cfbbe553e23b895e27956588",
    "graphbreaks": "55f772f77864752f2e98a6fc7713b45a1843c167",
    "codesign": "50cb459d080bcd783a4b481d3bde4150d35ac497",
    "sdpa": "22f654553bbe7aa28337ce34a25f1a9d27cee111",
    "sdpa-decoder": "7dc75fdf283693f73606f2fe7fdcb693afcb16b9",
    "predict-masks-nested": "187e2359f9eb3b00d43487a1ec3db849964753e4",
    "use-rel-pos": "d2fa29d580eaf7928eef702cd71d133b943c30cf",
    "hacky-nested-encoder": "8f2fc3cc90b222a2431d4c43379282e36f021b69",
    "wip-flash-nested": "e01edb904a49c449425fca9e48902824b22cf764",
    "wip-flash-sdpa-decoder": "bb1c8b6f3749b1a5f31635f5d2f26bcafa9d94f9"}



def change_sam_commit(sam_path, commit_name):
    assert commit_name in sam_commits
    root_cmd = ["git", "-C", sam_path]
    result = subprocess.run(
        root_cmd + ["checkout", sam_commits[commit_name]], capture_output=True)
    assert result.returncode == 0
    result = subprocess.run(
        root_cmd + ["rev-parse", "HEAD"], capture_output=True)
    assert result.returncode == 0


def run_experiment(experiments_data,
                   sam_path,
                   model_type,
                   idx,
                   sam_commit_name,
                   batch_size=1,
                   num_workers=0,
                   use_half=None,
                   use_compile="False",
                   compress=None,
                   use_nested_tensor=False,
                   extra_args=None,
                   print_header=False,
                   capture_output=True,
                   limit=None,
                   profile_path=None,
                   profile_top=False,
                   memory_path=None):
    root_cmd = ["python", "eval_combo.py",
                "--coco_root_dir",
                f"{experiments_data}/datasets/coco2017",
                "--coco_slice_name",
                "val2017",
                "--sam_checkpoint_base_path",
                f"{experiments_data}/checkpoints",
                "--sam_model_type",
                "vit_b",
                "--point_sampling_cache_dir",
                f"{experiments_data}/tmp/sam_coco_mask_center_cache",
                "--mask_debug_out_dir",
                f"{experiments_data}/tmp/sam_eval_masks_out"]
    args = root_cmd
    args = args + ["--sam_model_type", model_type]
    args = args + ["--batch_size", str(batch_size)]
    args = args + ["--num_workers", str(num_workers)]
    args = args + ["--use_compile", use_compile]
    if sam_commit_name == "local-fork":
        args = args + ["--use_local_sam_fork", "True"]
    else:
        change_sam_commit(sam_path, sam_commit_name)
    if use_half:
        args = args + ["--use_half", use_half]
    if compress is not None:
        args = args + ["--compress", compress]
    if use_nested_tensor:
        args = args + ["--use_nested_tensor", str(use_nested_tensor)]
    if limit is not None:
        args = args + ["--limit", str(limit)]
    if profile_path is not None:
        args = args + ["--profile-path", profile_path]
    if profile_top:
        args = args + ["--profile-top", "True"]
    if memory_path is not None:
        args = args + ["--memory-path", memory_path]
    if extra_args is None:
        extra_args = []
    args = args + extra_args
    if print_header:
        args = args + ["--print_header", "True"]
    import time
    t0 = time.time()
    result = subprocess.run(args, capture_output=capture_output)
    if not capture_output:
        return
    t1 = time.time()
    import torch
    pytorch_version = torch.__version__
    prefix = ",".join(
        map(str, [idx, (t1 - t0)/60.0, sam_commit_name, pytorch_version]))
    if result.returncode != 0:
        print(prefix + ",ERROR")
        return
    if print_header:
        header = result.stdout.decode().split("\n")[-3]
        print("technique,time,sam_commit_name,pytorch_version," + header)
    print(prefix + "," + result.stdout.decode().split("\n")[-2])


def run_traces_fn(traces_dir, pytorch_path, rexp, *args, **kwargs):
    # Limit to 10 batches
    kwargs['limit'] = 160

    # Create kernel traces
    profile_path = f"{traces_dir}/{args[0]}.json.gz"
    kwargs['profile_path'] = profile_path
    rexp(*args, **kwargs)
    kwargs['profile_path'] = None

    # Don't print header again if already printed
    kwargs['print_header'] = False

    # Create memory trace
    if 'use_compile' in kwargs and kwargs['use_compile'] == "max-autotune":
        # Memory traces don't seem to support CUDA graphs
        kwargs['use_compile'] = "max-autotune-no-cudagraphs"

    memory_path = f"{traces_dir}/{args[0]}"
    kwargs['memory_path'] = memory_path + ".pickle"
    rexp(*args, **kwargs)
    kwargs['memory_path'] = None

    # Convert memory trace to html page
    conversion_cmd = ["python", f"{pytorch_path}/torch/cuda/_memory_viz.py",
                      "trace_plot", memory_path + ".pickle", "-o", memory_path + ".html"]
    result = subprocess.run(conversion_cmd, capture_output=True)

def run(batch_size,
        model,
        pytorch_path,
        sam_path,
        experiments_data,
        run_traces=False,
        run_experiments=False,
        traces_dir=None,
        num_workers=32,
        print_header=True,
        capture_output=True,
        local_fork_only=False):

    assert model == "vit_b" or model == "vit_h"

    rexp = functools.partial(run_experiment,
                             experiments_data,
                             sam_path,
                             model,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             capture_output=capture_output)

    print_header = True
    if run_traces:
        assert traces_dir is not None
        rt = functools.partial(run_traces_fn, traces_dir, pytorch_path, rexp)

        if local_fork_only:
            rt("fp32",       "local-fork",   print_header=print_header)
            rt("fp16",       "local-fork",   use_half="bfloat16")
            rt("compile",    "local-fork",   use_half="bfloat16",  use_compile="max-autotune")
            # The local fork already uses SDPA + Triton for all of the above experiments.
            # local_fork_only mainly exists to ablate the order in which we apply
            # techniques and cannot be used to reproduce the experimental results
        else:
            rt("fp32",       "default",      print_header=print_header)
            rt("fp16",       "codesign",     use_half="bfloat16")
            rt("compile",    "codesign",     use_half="bfloat16",  use_compile="max-autotune")
            rt("SDPA",       "sdpa-decoder", use_half="bfloat16",  use_compile="max-autotune")
            rt("Triton",     "local-fork",   use_half="bfloat16",  use_compile="max-autotune")
        if batch_size > 1:
            rt("NT",         "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=True)
        rt("int8",           "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=True, compress="dynamic_quant")
        rt("sparse",         "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=True, compress="sparse")

    if run_experiments:
        if local_fork_only:
            rexp("fp32",     "local-fork",     print_header=print_header)
            rexp("bf16",     "local-fork",     use_half="bfloat16")
            rexp("compile",  "local-fork",     use_half="bfloat16",  use_compile="max-autotune")
            # The local fork already uses SDPA + Triton for all of the above experiments.
            # local_fork_only mainly exists to ablate the order in which we apply
            # techniques and cannot be used to reproduce the experimental results
        else:
            rexp("fp32",     "default",      print_header=print_header)
            rexp("bf16",     "codesign",     use_half="bfloat16")
            rexp("compile",  "codesign",     use_half="bfloat16",  use_compile="max-autotune")
            rexp("SDPA",     "sdpa-decoder", use_half="bfloat16",  use_compile="max-autotune")
            rexp("Triton",   "local-fork",   use_half="bfloat16",  use_compile="max-autotune")
        if batch_size > 1:
            rexp("NT",       "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(batch_size > 1))
        rexp("int8",         "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(batch_size > 1), compress="dynamic_quant")
        rexp("sparse",       "local-fork",   use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(batch_size > 1), compress="sparse")


if __name__ == '__main__':
    fire.Fire(run)
