#!/usr/bin/env python3
"""
PyTorch Native 实现的 Fused Sigmoid Gating Delta Rule Update

这个实现完全复刻 Triton kernel 的逻辑，用于：
1. 跨平台兼容（CPU, GPU, 自定义加速器）
2. 调试和验证
3. 作为 Triton 版本不可用时的 fallback
"""

import torch
from typing import Optional


def fused_sigmoid_gating_delta_rule_update_native(
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T, HV] or [1, total_len, HV] if varlen
    dt_bias: torch.Tensor,  # [HV]
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,  # [B, T, H, K] or [1, total_len, H, K] if varlen
    k: torch.Tensor,  # [B, T, H, K] or [1, total_len, H, K] if varlen
    v: torch.Tensor,  # [B, T, HV, V] or [1, total_len, HV, V] if varlen
    b: torch.Tensor,  # [B, T, HV] or [1, total_len, HV] if varlen
    initial_state_source: Optional[torch.Tensor],  # [num_states, HV, K, V]
    initial_state_indices: Optional[torch.Tensor],  # [N]
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,  # [N+1] for N sequences
):
    """
    PyTorch Native 实现的 Fused Sigmoid Gating Delta Rule Update

    这是一个循环神经网络的变体，结合了：
    - Sigmoid gating（门控）
    - Delta rule update（增量规则更新）
    - Recurrent state（循环状态）

    参数:
        A_log: log(A) 参数，形状 [HV]
        a: 时间相关的 gating 参数
           - 固定长度: [B, T, HV]
           - 变长序列: [1, total_len, HV]，配合 cu_seqlens 使用
        dt_bias: 时间偏置，形状 [HV]
        softplus_beta: softplus 函数的 beta 参数
        softplus_threshold: softplus 数值稳定性阈值
        q: query
           - 固定长度: [B, T, H, K]
           - 变长序列: [1, total_len, H, K]
        k: key
           - 固定长度: [B, T, H, K]
           - 变长序列: [1, total_len, H, K]
        v: value
           - 固定长度: [B, T, HV, V]
           - 变长序列: [1, total_len, HV, V]
        b: sigmoid gating 参数
           - 固定长度: [B, T, HV]
           - 变长序列: [1, total_len, HV]
        initial_state_source: 初始状态池，形状 [num_states, HV, K, V] 或 None
        initial_state_indices: 每个序列的初始状态索引，形状 [N] 或 None
        scale: Q 的缩放因子（默认为 K^-0.5）
        use_qk_l2norm_in_kernel: 是否在 kernel 内进行 Q/K L2 归一化
        cu_seqlens: 变长序列的累积序列长度，形状 [N+1]
                   例如 [0, 5, 12, 20] 表示 3 个序列，长度分别为 5, 7, 8
                   如果为 None，则假设所有序列长度相同

    返回:
        o: 输出
           - 固定长度: [B, T, HV, V]
           - 变长序列: [1, total_len, HV, V]

    核心算法:
        for t in range(T):
            # 1. Compute gating
            g = -exp(A_log) * softplus(a[:, t] + dt_bias)
            beta = sigmoid(b[:, t])

            # 2. Apply L2 norm (optional)
            if use_qk_l2norm:
                q[:, t] = normalize(q[:, t])
                k[:, t] = normalize(k[:, t])

            # 3. Scale query
            q[:, t] *= scale

            # 4. Decay hidden state
            h *= exp(g)

            # 5. Delta rule: subtract projection
            v_adjusted = v[:, t] - einsum('hk,hkv->hv', k[:, t], h)

            # 6. Apply beta gating
            v_adjusted *= beta

            # 7. Update hidden state
            h += einsum('hk,hv->hkv', k[:, t], v_adjusted)

            # 8. Compute output
            o[:, t] = einsum('hk,hkv->hv', q[:, t], h)
    """
    # 参数检查和初始化
    B, T, H, K = k.shape
    _, _, HV, V = v.shape

    # 判断是否为变长序列模式
    is_varlen = cu_seqlens is not None

    if is_varlen:
        # 变长序列模式：cu_seqlens = [0, len1, len1+len2, ...]
        # 输入形状应该是 [1, total_len, ...]
        assert B == 1, f"Variable length mode requires B=1, got B={B}"
        N = len(cu_seqlens) - 1  # 序列数量
        total_len = T
    else:
        # 固定长度模式：每个序列长度都是 T
        N = B
        total_len = B * T

    # 默认 scale
    if scale is None:
        scale = K ** -0.5

    # 转换为 float32 进行计算（与 Triton 一致）
    compute_dtype = torch.float32
    original_dtype = q.dtype

    A_log = A_log.to(compute_dtype)
    dt_bias = dt_bias.to(compute_dtype)
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)

    # 处理 a 和 b 的形状：自动添加 batch 维度（如果缺失）
    # 期望形状：[B, T, HV] 或 [1, total_len, HV]（变长模式）
    # 实际可能：[T, HV] 或 [total_len, HV]
    if a.dim() == 2:
        # 缺少 batch 维度，添加它
        a = a.unsqueeze(0)  # [T, HV] -> [1, T, HV]
    if b.dim() == 2:
        # 缺少 batch 维度，添加它
        b = b.unsqueeze(0)  # [T, HV] -> [1, T, HV]

    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    # 输出缓冲区
    if is_varlen:
        o = torch.zeros(1, total_len, HV, V,
                        dtype=compute_dtype, device=q.device)
    else:
        o = torch.zeros(B, T, HV, V, dtype=compute_dtype, device=q.device)

    # 处理每个序列
    for seq_idx in range(N):
        # 获取当前序列的起始和结束位置
        if is_varlen:
            bos = cu_seqlens[seq_idx].item()  # begin of sequence
            eos = cu_seqlens[seq_idx + 1].item()  # end of sequence
            seq_len = eos - bos
        else:
            bos = seq_idx * T
            eos = bos + T
            seq_len = T

        # 初始化 hidden state: [HV, K, V]
        h = torch.zeros(HV, K, V, dtype=compute_dtype, device=q.device)

        # 加载初始状态（如果提供）
        if initial_state_source is not None and initial_state_indices is not None:
            state_idx = initial_state_indices[seq_idx].item()
            if state_idx >= 0:
                h = initial_state_source[state_idx].clone().to(compute_dtype)

        # 处理当前序列的每个时间步
        for t_rel in range(seq_len):
            # 计算在输入张量中的绝对位置
            t_abs = bos + t_rel

            # 1. 加载当前时间步的输入
            if is_varlen:
                q_t = q[0, t_abs]  # [H, K]
                k_t = k[0, t_abs]  # [H, K]
                v_t = v[0, t_abs]  # [HV, V]
                b_t = b[0, t_abs]  # [HV]
                a_t = a[0, t_abs]  # [HV]
            else:
                q_t = q[seq_idx, t_rel]  # [H, K]
                k_t = k[seq_idx, t_rel]  # [H, K]
                v_t = v[seq_idx, t_rel]  # [HV, V]
                b_t = b[seq_idx, t_rel]  # [HV]
                a_t = a[seq_idx, t_rel]  # [HV]

            # 2. 计算 sigmoid gating 参数
            # g = -exp(A_log) * softplus(a + dt_bias)
            x = a_t + dt_bias  # [HV]
            beta_x = softplus_beta * x

            # Softplus with numerical stability
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x
            )
            g = -torch.exp(A_log) * softplus_x  # [HV]

            # beta = sigmoid(b)
            beta = torch.sigmoid(b_t)  # [HV]

            # 3. 处理每个 head (HV)
            for hv_idx in range(HV):
                # 确定对应的 H index
                # 假设 HV >= H, 且 HV 是 H 的倍数
                h_idx = hv_idx // (HV // H) if HV >= H else 0

                # 获取当前 head 的 q, k
                q_h = q_t[h_idx]  # [K]
                k_h = k_t[h_idx]  # [K]
                v_h = v_t[hv_idx]  # [V]

                # 获取当前 head 的 hidden state
                h_hv = h[hv_idx]  # [K, V]

                # 4. Apply L2 normalization (optional)
                if use_qk_l2norm_in_kernel:
                    q_h = q_h / (torch.sqrt(torch.sum(q_h * q_h) + 1e-6))
                    k_h = k_h / (torch.sqrt(torch.sum(k_h * k_h) + 1e-6))

                # 5. Scale query
                q_h = q_h * scale

                # 6. Decay hidden state: h *= exp(g)
                h_hv = h_hv * torch.exp(g[hv_idx])

                # 7. Delta rule: v -= sum(h * k, dim=0)
                # h: [K, V], k: [K] -> h * k: [K, V]
                # sum over K dimension -> [V]
                v_adjusted = v_h - torch.sum(h_hv * k_h[:, None], dim=0)

                # 8. Apply beta gating: v *= beta
                v_adjusted = v_adjusted * beta[hv_idx]

                # 9. Update hidden state: h += k[:, None] * v[None, :]
                h_hv = h_hv + k_h[:, None] * v_adjusted[None, :]

                # 10. Compute output: o = sum(h * q, dim=0)
                o_hv = torch.sum(h_hv * q_h[:, None], dim=0)

                # 存储输出
                if is_varlen:
                    o[0, t_abs, hv_idx] = o_hv
                else:
                    o[seq_idx, t_rel, hv_idx] = o_hv

                # 更新 hidden state
                h[hv_idx] = h_hv

        # 保存最终状态（如果需要）
        if initial_state_source is not None and initial_state_indices is not None:
            state_idx = initial_state_indices[seq_idx].item()
            if state_idx >= 0:
                initial_state_source[state_idx] = h.to(
                    initial_state_source.dtype)

    # 转换回原始 dtype
    o = o.to(original_dtype)
    return o


def fused_sigmoid_gating_delta_rule_update_native_optimized(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    优化版本：使用向量化操作减少循环

    注意：这个版本假设 H == HV（简化情况）
    """
    B, T, H, K = k.shape
    _, _, HV, V = v.shape

    assert H == HV, "Optimized version requires H == HV"

    # 判断是否为变长序列模式
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1, f"Variable length mode requires B=1, got B={B}"
        N = len(cu_seqlens) - 1
        total_len = T
    else:
        N = B
        total_len = B * T

    if scale is None:
        scale = K ** -0.5

    compute_dtype = torch.float32
    original_dtype = q.dtype

    # 转换数据类型
    A_log = A_log.to(compute_dtype)
    dt_bias = dt_bias.to(compute_dtype)
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)

    # 处理 a 和 b 的形状：自动添加 batch 维度（如果缺失）
    if a.dim() == 2:
        a = a.unsqueeze(0)  # [T, HV] -> [1, T, HV]
    if b.dim() == 2:
        b = b.unsqueeze(0)  # [T, HV] -> [1, T, HV]

    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    # 输出
    if is_varlen:
        o = torch.zeros(1, total_len, HV, V,
                        dtype=compute_dtype, device=q.device)
    else:
        o = torch.zeros(B, T, HV, V, dtype=compute_dtype, device=q.device)

    # 处理每个序列
    for seq_idx in range(N):
        # 获取当前序列的起始和结束位置
        if is_varlen:
            bos = cu_seqlens[seq_idx].item()
            eos = cu_seqlens[seq_idx + 1].item()
            seq_len = eos - bos
        else:
            bos = seq_idx * T
            eos = bos + T
            seq_len = T

        # 初始化 hidden state: [HV, K, V]
        h = torch.zeros(HV, K, V, dtype=compute_dtype, device=q.device)

        # 加载初始状态
        if initial_state_source is not None and initial_state_indices is not None:
            state_idx = initial_state_indices[seq_idx].item()
            if state_idx >= 0:
                h = initial_state_source[state_idx].clone().to(compute_dtype)

        # 处理当前序列的每个时间步
        for t_rel in range(seq_len):
            t_abs = bos + t_rel

            # 加载输入
            if is_varlen:
                q_t = q[0, t_abs]  # [H, K]
                k_t = k[0, t_abs]  # [H, K]
                v_t = v[0, t_abs]  # [HV, V]
                b_t = b[0, t_abs]  # [HV]
                a_t = a[0, t_abs]  # [HV]
            else:
                q_t = q[seq_idx, t_rel]  # [H, K]
                k_t = k[seq_idx, t_rel]  # [H, K]
                v_t = v[seq_idx, t_rel]  # [HV, V]
                b_t = b[seq_idx, t_rel]  # [HV]
                a_t = a[seq_idx, t_rel]  # [HV]

            # 计算 gating
            x = a_t + dt_bias
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x
            )
            g = -torch.exp(A_log) * softplus_x  # [HV]
            beta = torch.sigmoid(b_t)  # [HV]

            # L2 normalization
            if use_qk_l2norm_in_kernel:
                q_t = q_t / \
                    (torch.sqrt((q_t ** 2).sum(dim=-1, keepdim=True)) + 1e-6)
                k_t = k_t / \
                    (torch.sqrt((k_t ** 2).sum(dim=-1, keepdim=True)) + 1e-6)

            # Scale query
            q_t = q_t * scale

            # Decay hidden state: h *= exp(g) [HV, K, V]
            h = h * torch.exp(g)[:, None, None]

            # Delta rule: v -= einsum('hk,hkv->hv', k, h)
            # h: [HV, K, V], k: [H, K] -> 对于每个 HV, 计算 sum_k(h[hv, k, :] * k[hv, k])
            v_adjusted = v_t - torch.einsum('hk,hkv->hv', k_t, h)  # [HV, V]

            # Apply beta gating
            v_adjusted = v_adjusted * beta[:, None]

            # Update hidden state: h += einsum('hk,hv->hkv', k, v_adjusted)
            h = h + torch.einsum('hk,hv->hkv', k_t, v_adjusted)

            # Compute output: o = einsum('hk,hkv->hv', q, h)
            output = torch.einsum('hk,hkv->hv', q_t, h)
            if is_varlen:
                o[0, t_abs] = output
            else:
                o[seq_idx, t_rel] = output

        # 保存最终状态
        if initial_state_source is not None and initial_state_indices is not None:
            state_idx = initial_state_indices[seq_idx].item()
            if state_idx >= 0:
                initial_state_source[state_idx] = h.to(
                    initial_state_source.dtype)

    return o.to(original_dtype)


if __name__ == "__main__":
    print("=" * 80)
    print("Fused Sigmoid Gating Delta Rule Update - Native Implementation")
    print("=" * 80)
    print("\n✅ 实现完成！")
    print("\n包含的函数：")
    print("  1. fused_sigmoid_gating_delta_rule_update_native()          - 完整实现（逐 head 处理）")
    print("  2. fused_sigmoid_gating_delta_rule_update_native_optimized() - 优化实现（向量化）")
    print("\n运行测试：python3 test_fused_sigmoid_gating_native.py")
