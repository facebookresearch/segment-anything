"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

This version was modified to fuse an addition of two attention masks into one
attn_bias = (rel_h_ + rel_w_).view(q_.size(0), q_.size(1), rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4))

We use attn_mask and attn_bias interchangeably.

This modification was designed by Christian Puhrsch and Daniel Haziza

"""

import torch

import triton
import triton.language as tl

import os
import pathlib


@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vk, stride_vn,
    stride_oh, stride_om, stride_on,
    stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    OUT_DTYPE: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)  # , boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_hz * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :])
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # , boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr)
        # , boundary_check=(1, 0), padding_option="zero")
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N)
        qk += (b0 + b1)

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE))


def _autotune(configs, function):
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
            t0 = benchmark.Timer(
                stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
            )
        except:
            return None
        return t0.blocked_autorange().mean * 1e6

    best = None
    best_config = None
    for config in configs:
        BLOCK_M, BLOCK_N, num_warps, num_stages = config
        t_config = benchmark_torch_function_in_microseconds(
            function, BLOCK_M, BLOCK_N, num_warps, num_stages)
        if t_config is not None:
            if best is not None:
                if t_config < best:
                    best = t_config
                    best_config = config
            else:
                best = t_config
                best_config = config
        print(str(config), " :", str(t_config))
    return best, best_config


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 num_warps,
                                                 num_stages):
    _, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    assert q.size() == k.size()
    assert q.size() == v.size()
    assert q.size(-2) == rel_h_w.size(-2)
    assert (q.dtype == torch.bfloat16 or q.dtype == torch.float16)
    assert k.dtype == q.dtype
    assert v.dtype == k.dtype
    assert o.dtype == v.dtype
    assert rel_h_w.dtype == q.dtype
    assert rel_h_w.size(-1) == 128
    # assert rel_h_w.size(-1) == 2 * BLOCK_N

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # print("q.shape[0] * q.shape[1]: ", q.shape[0] * q.shape[1])
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    assert P_SEQ == 0
    assert rel_h_w.is_contiguous(), str(rel_h_w.stride())
    _fwd_kernel_aligned[grid](
        q, k, v,
        rel_h_w,
        sm_scale,
        o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        rel_h_w.stride(1), rel_h_w.stride(2),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        P_SEQ,
        OUT_DTYPE=tl.float16 if q.dtype == torch.float16 else tl.bfloat16,
        BIAS_LAST_SIZE=(rel_h_w.size(-1) // 2),
        B0_NUMEL=rel_h_w.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=num_stages)


def _load_best_configs():
    device_name = torch.cuda.get_device_name()
    if not device_name.startswith('NVIDIA A100'):
        print("Warning: Custom flash attention kernels were written specifically for A100.")
    import importlib
    saved_configs = importlib.resources.files("segment_anything_fast")
    saved_configs = saved_configs / "configs" / "flash_4_configs_a100.p"
    if not device_name.startswith('NVIDIA A100'):
        cwd = pathlib.Path.cwd()
        saved_configs = cwd / "flash_4_configs.p"
        print(f"We will try to read previously created kernel configurations from {saved_configs}.")
        print("You can disable this kernel by setting SEGMENT_ANYTHING_FAST_USE_FLASH_4=0")
        return None
    if saved_configs.is_file():
        import pickle
        with open(saved_configs, 'rb') as f:
            print(f"Loading best configs from file {saved_configs}")
            return pickle.load(f)


def _save_best_configs(best_configs):
    import importlib
    saved_configs = importlib.resources.files("segment_anything_fast")
    saved_configs = saved_configs / "configs" / "flash_4_configs_a100.p"
    device_name = torch.cuda.get_device_name()
    if not device_name.startswith('NVIDIA A100'):
        saved_configs = pathlib.Path.cwd() / "flash_4_configs.p"
        print("Warning: Custom flash attention kernels were written specifically for A100.")
        print(f"Storing configs for {device_name} locally under {saved_configs}")
    with open(saved_configs, 'wb') as f:
        import pickle
        print(f"Saving best configs to file {saved_configs}")
        pickle.dump(best_configs, f)


def _create_best_configs_key(q, k, v, rel_h_w, o):
    key = (q.size(),   k.size(),   v.size(),   rel_h_w.size(),   o.size(),
           q.stride(), k.stride(), v.stride(), rel_h_w.stride(), o.stride())
    return key


BEST_CONFIGS = None

lib = torch.library.Library("customflash", "FRAGMENT")
lib.define("custom_flash_aligned(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor")


# All that's needed for torch.compile support
@torch.library.impl(lib, "custom_flash_aligned", "Meta")
def _attention_rel_h_rel_w_kernel_aligned_meta(q, k, v, rel_h_w, sm_scale):
    return q.contiguous()


@torch.library.impl(lib, "custom_flash_aligned", "CUDA")
def _attention_rel_h_rel_w_kernel_aligned(q, k, v, rel_h_w, sm_scale):
    # This is likely not needed, but without it the kernel
    # is guaranteed to fail. If the inputs are already contiguous
    # these are cheap checks via is_contiguous and do nothing.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q, memory_format=torch.contiguous_format)

    global BEST_CONFIGS
    if BEST_CONFIGS is None:
        BEST_CONFIGS = _load_best_configs()
    # Loading must have not been successful. Let's create a new dictionary.
    if BEST_CONFIGS is None:
        BEST_CONFIGS = {}
    key = _create_best_configs_key(q, k, v, rel_h_w, o)
    if key not in BEST_CONFIGS:
        print("key ", key, " not found. Running autotune. This might take a while.")
        import functools
        import itertools
        configs = []
        for (BLOCK_M, BLOCK_N, num_warps) in itertools.product([64, 128], [64, 128], [1, 2, 4, 8]):
            for num_stages in range(1, num_warps + 1):
                configs.append((BLOCK_M, BLOCK_N, num_warps, num_stages))
        print("all configs len: ", len(configs))
        best, best_config = _autotune(configs, functools.partial(_attention_rel_h_rel_w_kernel_aligned_device,
                                                                 q, k, v, rel_h_w, sm_scale, o))
        BEST_CONFIGS[key] = best_config
        print("Found best_config ", best_config,
              " with time ", best, " for key ", key)
        _save_best_configs(BEST_CONFIGS)
    best_config = BEST_CONFIGS[key]
    if best_config is None:
        return torch.tensor([])

    _attention_rel_h_rel_w_kernel_aligned_device(q,
                                                 k,
                                                 v,
                                                 rel_h_w,
                                                 sm_scale,
                                                 o,
                                                 best_config[0],
                                                 best_config[1],
                                                 best_config[2],
                                                 best_config[3])

    return o


USE_CUSTOM_KERNEL = bool(int(os.environ.get('SEGMENT_ANYTHING_FAST_USE_FLASH_4', 1)))


def _attention_rel_h_rel_w(q_, k_, v_, rel_h_, rel_w_):
    """
    Writing this as a composite allows torch.compile to fuse
    the needed padding into previous operations and memory
    allocations.
    """

    import math
    sm_scale = 1. / math.sqrt(q_.size(-1))
    # Check if second last dimension is multiple of 256
    q_size_2_padded = (((q_.size(-2) + 256 - 1) // 256) * 256) - q_.size(-2)

    def kernel_guards(q_, k_, v_):
        return (q_.dtype == torch.bfloat16 or q_.dtype == torch.float16) and q_.dtype == k_.dtype and k_.dtype == v_.dtype and USE_CUSTOM_KERNEL
    # vit_b and vit_l
    if q_size_2_padded == 0 and q_.size(-1) == 64 and kernel_guards(q_, k_, v_):
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.customflash.custom_flash_aligned(
            q_, k_, v_, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o
    # vit_h
    if q_size_2_padded == 0 and q_.size(-1) == 80 and kernel_guards(q_, k_, v_):
        # Only support multiples of 64, so need to pad
        q = torch.nn.functional.pad(q_, (0, 128 - 80, 0, 0), "constant", 0)
        k = torch.nn.functional.pad(k_, (0, 128 - 80, 0, 0), "constant", 0)
        v = torch.nn.functional.pad(v_, (0, 128 - 80, 0, 0), "constant", 0)
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.customflash.custom_flash_aligned(
            q, k, v, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o[:, :, :, :80]
    attn_bias = (rel_h_ + rel_w_).view(q_.size(0), q_.size(1),
                                       rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4))
    return torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_bias)
