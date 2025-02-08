# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = (["Flash2", "Pytorch"]
           + (["Triton"] if attention_triton is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else []))

methods = (["Flash2_full", "Flash2_local"]) 
local_seq_len = 128

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
flops_config = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            
            ### flash_attn_qkvpacked_func
            
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            f, b = time_fwd_bwd(
                flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            # time_f[config, "Flash2"] = f
            # time_b[config, "Flash2"] = b
            time_f[config, "Flash2_full"] = f
            time_b[config, "Flash2_full"] = b
            flops_config[config, "Flash2_full"] = (batch_size, seqlen, headdim, nheads, causal)
            
            local_chunked_qkv = qkv.reshape(-1, local_seq_len, 3, nheads, headdim)
            f, b = time_fwd_bwd(
                flash_attn_qkvpacked_func, local_chunked_qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "Flash2_local"] = f
            time_b[config, "Flash2_local"] = b
            flops_config[config, "Flash2_local"] = (local_chunked_qkv.shape[0], local_seq_len, headdim, nheads, causal)
            
            ### flash_attn_qkvpacked_func
            
            # ### flash_attn_varlen_func
            # q = qkv[:, :, 0]
            # k = qkv[:, :, 1]
            # v = qkv[:, :, 2]
            # q = q.view(-1, nheads, headdim)
            # k = k.view(-1, nheads, headdim)
            # v = v.view(-1, nheads, headdim)
            # cu_seqlens_q = torch.tensor([seqlen] * qkv.shape[0], device=device, dtype=torch.int32)
            # cu_seqlens_q = torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.int32)
            # cu_seqlens_q = F.pad(cu_seqlens_q, (1, 0), value=0)
            # max_seqlen_q = seqlen
            # max_seqlen_k = seqlen
            # cu_seqlens_k = cu_seqlens_q
            
            # # print(cu_seqlens_q)
            # f, b = time_fwd_bwd(
            #     flash_attn_varlen_func, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, causal=causal, repeats=repeats, verbose=False
            # )
            # time_f[config, "Flash2_full"] = f
            # time_b[config, "Flash2_full"] = b
            # flops_config[config, "Flash2_full"] = (batch_size, seqlen, headdim, nheads, causal)
            
            # local_chunked_qkv = qkv.view(-1, local_seq_len, 3, nheads, headdim)
            # local_chunked_q = local_chunked_qkv[:, :, 0]
            # local_chunked_k = local_chunked_qkv[:, :, 1]
            # local_chunked_v = local_chunked_qkv[:, :, 2]
            # local_chunked_q = local_chunked_q.view(-1, nheads, headdim)
            # local_chunked_k = local_chunked_k.view(-1, nheads, headdim)
            # local_chunked_v = local_chunked_v.view(-1, nheads, headdim)
            # cu_seqlens_q = torch.tensor([local_seq_len] * local_chunked_qkv.shape[0], device=device, dtype=torch.int32)
            # cu_seqlens_q = torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.int32)
            # cu_seqlens_q = F.pad(cu_seqlens_q, (1, 0), value=0)
            # max_seqlen_q = local_seq_len
            # max_seqlen_k = local_seq_len
            # cu_seqlens_k = cu_seqlens_q
            
            # # print(cu_seqlens_q)
            # f, b = time_fwd_bwd(
            #     flash_attn_varlen_func, local_chunked_q, local_chunked_k, local_chunked_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, causal=causal, repeats=repeats, verbose=False
            # )
            # time_f[config, "Flash2_local"] = f
            # time_b[config, "Flash2_local"] = b
            # flops_config[config, "Flash2_local"] = (local_chunked_qkv.shape[0], local_seq_len, headdim, nheads, causal)
            
            # ### flash_attn_qkvpacked_func
            


            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen}, qkv.shape={qkv.shape}, local_chunked_qkv.shape={local_chunked_qkv.shape} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                method_flops_config = flops_config[config, method]
                speed_f[config, method] = efficiency(
                    # flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    flops(*method_flops_config, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    # flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    flops(*method_flops_config, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    # flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    flops(*method_flops_config, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                    f" (fwd: {time_f[config, method]*1000:.3f}ms, "
                    f"bwd: {time_b[config, method]*1000:.3f}ms)"
                )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
