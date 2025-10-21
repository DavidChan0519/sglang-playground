#!/usr/bin/env python3
"""
PyTorch Native 实现的 LayerNorm（等价于 Triton kernel）

这个实现完全复刻 _layer_norm_fwd_1pass_kernel 的逻辑，包括：
1. LayerNorm / RMSNorm
2. GroupNorm 支持
3. SwiGLU Gating（门控前/后）
4. 可选 bias
"""

import torch
import torch.nn.functional as F


def _layer_norm_fwd_native(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """
    PyTorch Native 实现的 LayerNorm forward

    等价于 Triton kernel _layer_norm_fwd_1pass_kernel

    Args:
        x: 输入 tensor [M, N]
        weight: 权重 [N]
        bias: 偏置 [N] 或 None
        eps: epsilon for numerical stability
        z: 门控值 [M, N] 或 None
        out: 输出 tensor [M, N] 或 None（自动分配）
        group_size: GroupNorm 的组大小（默认 N，即标准 LayerNorm）
        norm_before_gate: True = 先 norm 后 gate，False = 先 gate 后 norm
        is_rms_norm: True = RMSNorm，False = LayerNorm

    Returns:
        out: 输出 [M, N]
        mean: 均值 [ngroups * M] 或 None (RMSNorm)
        rstd: 1/std [ngroups * M]
    """
    M, N = x.shape

    # 处理 group_size
    if group_size is None:
        group_size = N
    assert N % group_size == 0, f"N ({N}) must be divisible by group_size ({group_size})"
    ngroups = N // group_size

    # Ensure weight and bias are on the same device as x
    # This prevents device mismatch errors
    if weight.device != x.device:
        import warnings
        warnings.warn(
            f"LayerNorm Native: weight is on {weight.device} but x is on {x.device}. "
            f"Moving weight to {x.device}. This may indicate an upstream bug."
        )
        weight = weight.to(x.device)
    if bias is not None and bias.device != x.device:
        import warnings
        warnings.warn(
            f"LayerNorm Native: bias is on {bias.device} but x is on {x.device}. "
            f"Moving bias to {x.device}. This may indicate an upstream bug."
        )
        bias = bias.to(x.device)

    # 分配输出
    if out is None:
        out = torch.empty_like(x)

    # 转换为 float32 进行计算（与 Triton 一致）
    x_compute = x.to(torch.float32)
    weight_compute = weight.to(torch.float32)
    bias_compute = bias.to(torch.float32) if bias is not None else None
    z_compute = z.to(torch.float32) if z is not None else None

    # Reshape 为 [M, ngroups, group_size] 以支持 GroupNorm
    if ngroups > 1:
        x_reshaped = x_compute.view(M, ngroups, group_size)
        weight_reshaped = weight_compute.view(ngroups, group_size)
        bias_reshaped = bias_compute.view(
            ngroups, group_size) if bias_compute is not None else None
        z_reshaped = z_compute.view(
            M, ngroups, group_size) if z_compute is not None else None
    else:
        # 标准 LayerNorm（单组）
        x_reshaped = x_compute.unsqueeze(1)  # [M, 1, N]
        weight_reshaped = weight_compute.unsqueeze(0)  # [1, N]
        bias_reshaped = bias_compute.unsqueeze(
            0) if bias_compute is not None else None
        z_reshaped = z_compute.unsqueeze(1) if z_compute is not None else None

    # ============ Step 1: 可选的 Gating BEFORE normalization ============
    if z_reshaped is not None and not norm_before_gate:
        # SwiGLU: x = x * z * sigmoid(z)
        x_reshaped = x_reshaped * z_reshaped * torch.sigmoid(z_reshaped)

    # ============ Step 2: 计算统计量 ============
    if not is_rms_norm:
        # LayerNorm: 计算 mean 和 var
        mean = x_reshaped.mean(dim=2, keepdim=True)  # [M, ngroups, 1]
        xbar = x_reshaped - mean
        var = (xbar ** 2).mean(dim=2, keepdim=True)  # [M, ngroups, 1]

        # 存储 mean（flatten 为 [ngroups * M]）
        mean_out = mean.squeeze(2).t().contiguous().view(-1)  # [ngroups * M]
    else:
        # RMSNorm: 不计算 mean，直接计算 RMS
        mean = None
        mean_out = None
        xbar = x_reshaped
        var = (xbar ** 2).mean(dim=2, keepdim=True)  # [M, ngroups, 1]

    # 计算 rstd = 1 / sqrt(var + eps)
    rstd = 1.0 / torch.sqrt(var + eps)  # [M, ngroups, 1]

    # 存储 rstd（flatten 为 [ngroups * M]）
    rstd_out = rstd.squeeze(2).t().contiguous().view(-1)  # [ngroups * M]

    # ============ Step 3: 归一化 ============
    if not is_rms_norm:
        x_hat = xbar * rstd  # (x - mean) * rstd
    else:
        x_hat = x_reshaped * rstd  # x * rstd

    # ============ Step 4: 仿射变换 ============
    # y = x_hat * weight + bias
    y = x_hat * weight_reshaped.unsqueeze(0)  # [M, ngroups, group_size]
    if bias_reshaped is not None:
        y = y + bias_reshaped.unsqueeze(0)

    # ============ Step 5: 可选的 Gating AFTER normalization ============
    if z_reshaped is not None and norm_before_gate:
        # SwiGLU: y = y * z * sigmoid(z)
        y = y * z_reshaped * torch.sigmoid(z_reshaped)

    # ============ Step 6: Reshape 回原始形状 ============
    y = y.view(M, N)

    # 转换回原始 dtype
    out.copy_(y.to(x.dtype))

    return out, mean_out, rstd_out


def layernorm_fn_native(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    """
    用户接口：PyTorch Native LayerNorm

    等价于 layernorm_fn 但使用 native PyTorch 实现
    """
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    y, _, _ = _layer_norm_fwd_native(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )

    return y.reshape(x_shape_og)


def rmsnorm_fn_native(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True
):
    """
    用户接口：PyTorch Native RMSNorm
    """
    return layernorm_fn_native(
        x,
        weight,
        bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )


# ============================================================================
# 简化版本（仅标准 LayerNorm，无额外功能）
# ============================================================================

def simple_layernorm_native(x, weight, bias, eps=1e-6):
    """
    最简单的 PyTorch Native LayerNorm

    等价于 torch.nn.LayerNorm 但手动实现
    """
    # Ensure weight and bias are on the same device as x
    if weight.device != x.device:
        weight = weight.to(x.device)
    if bias is not None and bias.device != x.device:
        bias = bias.to(x.device)

    # 计算均值和方差
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

    # 归一化
    x_hat = (x - mean) / torch.sqrt(var + eps)

    # 仿射变换
    y = x_hat * weight + bias

    return y


if __name__ == "__main__":
    print("=" * 80)
    print("PyTorch Native LayerNorm Implementation")
    print("=" * 80)
    print("\n✅ 实现完成！")
    print("\n包含的函数：")
    print("  1. _layer_norm_fwd_native()      - 底层实现（完整功能）")
    print("  2. layernorm_fn_native()          - 用户接口（LayerNorm）")
    print("  3. rmsnorm_fn_native()            - 用户接口（RMSNorm）")
    print("  4. simple_layernorm_native()      - 简化版本（仅标准 LayerNorm）")
    print("\n运行测试：python3 test_layernorm_native.py")
