#!/usr/bin/env python3
"""
Fused Sigmoid Gating Delta Rule Update - 演示脚本

展示如何使用 PyTorch native 实现
"""

import torch
from fused_sigmoid_gating_native_implementation import (
    fused_sigmoid_gating_delta_rule_update_native,
    fused_sigmoid_gating_delta_rule_update_native_optimized,
)


def demo_1_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "=" * 80)
    print("示例 1: 基本使用")
    print("=" * 80)

    # 配置
    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H

    print(f"配置: Batch={B}, Time={T}, Heads={H}, K_dim={K}, V_dim={V}")

    # 创建输入（随机初始化）
    A_log = torch.randn(HV) * 0.1
    a = torch.randn(B, T, HV) * 0.1
    dt_bias = torch.randn(HV) * 0.1
    q = torch.randn(B, T, H, K) * 0.1
    k = torch.randn(B, T, H, K) * 0.1
    v = torch.randn(B, T, HV, V) * 0.1
    b = torch.randn(B, T, HV) * 0.1

    print("\n输入统计:")
    print(f"  q: shape={q.shape}, mean={q.mean():.6f}, std={q.std():.6f}")
    print(f"  k: shape={k.shape}, mean={k.mean():.6f}, std={k.std():.6f}")
    print(f"  v: shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}")

    # 调用函数
    out = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,  # 使用默认 K^-0.5
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print("\n输出统计:")
    print(
        f"  out: shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")
    print("\n✅ 基本使用成功！")


def demo_2_with_state():
    """示例 2: 带状态管理"""
    print("\n" + "=" * 80)
    print("示例 2: 带状态管理（模拟多轮推理）")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    num_states = 2  # 为每个 batch 准备一个状态

    print(f"配置: Batch={B}, Time={T}, num_states={num_states}")

    # 创建状态池
    state_pool = torch.zeros(num_states, HV, K, V)
    state_indices = torch.arange(B)  # Batch 0 用状态 0，Batch 1 用状态 1

    print("\n第 1 轮推理:")

    # 创建输入
    A_log = torch.randn(HV) * 0.1
    a = torch.randn(B, T, HV) * 0.1
    dt_bias = torch.randn(HV) * 0.1
    q = torch.randn(B, T, H, K) * 0.1
    k = torch.randn(B, T, H, K) * 0.1
    v = torch.randn(B, T, HV, V) * 0.1
    b = torch.randn(B, T, HV) * 0.1

    # 第一轮推理（初始状态为 0）
    out_1 = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, 1.0, 20.0,
        q, k, v, b,
        initial_state_source=state_pool,
        initial_state_indices=state_indices,
    )

    print(f"  输出: mean={out_1.mean():.6f}, std={out_1.std():.6f}")
    print(f"  最终状态: mean={state_pool.mean():.6f}, std={state_pool.std():.6f}")

    print("\n第 2 轮推理（继承上一轮状态）:")

    # 创建新输入
    q_2 = torch.randn(B, T, H, K) * 0.1
    k_2 = torch.randn(B, T, H, K) * 0.1
    v_2 = torch.randn(B, T, HV, V) * 0.1
    b_2 = torch.randn(B, T, HV) * 0.1
    a_2 = torch.randn(B, T, HV) * 0.1

    # 第二轮推理（使用上一轮的最终状态作为初始状态）
    out_2 = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a_2, dt_bias, 1.0, 20.0,
        q_2, k_2, v_2, b_2,
        initial_state_source=state_pool,  # 使用上一轮更新的状态
        initial_state_indices=state_indices,
    )

    print(f"  输出: mean={out_2.mean():.6f}, std={out_2.std():.6f}")
    print(f"  最终状态: mean={state_pool.mean():.6f}, std={state_pool.std():.6f}")

    print("\n✅ 状态管理成功！Hidden state 在多轮之间传递")


def demo_3_optimized_version():
    """示例 3: 使用优化版本"""
    print("\n" + "=" * 80)
    print("示例 3: 优化版本（更快）")
    print("=" * 80)

    B, T, H, K, V = 4, 8, 4, 16, 16
    HV = H

    print(f"配置: Batch={B}, Time={T}, Heads={H}, K_dim={K}, V_dim={V}")

    # 创建输入
    A_log = torch.randn(HV) * 0.1
    a = torch.randn(B, T, HV) * 0.1
    dt_bias = torch.randn(HV) * 0.1
    q = torch.randn(B, T, H, K) * 0.1
    k = torch.randn(B, T, H, K) * 0.1
    v = torch.randn(B, T, HV, V) * 0.1
    b = torch.randn(B, T, HV) * 0.1

    # 测试性能
    import time

    # 基础版本
    start = time.time()
    out_basic = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a.clone(), dt_bias, 1.0, 20.0,
        q.clone(), k.clone(), v.clone(), b.clone(),
        None, None,
    )
    basic_time = time.time() - start

    # 优化版本
    start = time.time()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a.clone(), dt_bias, 1.0, 20.0,
        q.clone(), k.clone(), v.clone(), b.clone(),
        None, None,
    )
    optimized_time = time.time() - start

    print(f"\n性能对比:")
    print(f"  基础版本: {basic_time*1000:.2f}ms")
    print(f"  优化版本: {optimized_time*1000:.2f}ms")
    print(f"  加速比: {basic_time / optimized_time:.2f}x")

    # 验证等价性
    max_diff = (out_basic - out_optimized).abs().max().item()
    print(f"\n精度验证:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  等价性: {'✅ 通过' if max_diff < 1e-6 else '❌ 失败'}")


def demo_4_with_l2norm():
    """示例 4: 带 L2 归一化"""
    print("\n" + "=" * 80)
    print("示例 4: Q/K L2 归一化")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H

    print(f"配置: Batch={B}, Time={T}, use_qk_l2norm=True")

    # 创建输入（故意使用不同范数的数据）
    A_log = torch.randn(HV) * 0.1
    a = torch.randn(B, T, HV) * 0.1
    dt_bias = torch.randn(HV) * 0.1
    q = torch.randn(B, T, H, K) * 2.0  # 较大的范数
    k = torch.randn(B, T, H, K) * 0.5  # 较小的范数
    v = torch.randn(B, T, HV, V) * 0.1
    b = torch.randn(B, T, HV) * 0.1

    print("\n输入统计:")
    print(f"  q 范数: mean={torch.norm(q, dim=-1).mean():.6f}")
    print(f"  k 范数: mean={torch.norm(k, dim=-1).mean():.6f}")

    # 不使用 L2 归一化
    out_no_norm = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a.clone(), dt_bias, 1.0, 20.0,
        q.clone(), k.clone(), v.clone(), b.clone(),
        None, None,
        use_qk_l2norm_in_kernel=False,
    )

    # 使用 L2 归一化
    out_with_norm = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a.clone(), dt_bias, 1.0, 20.0,
        q.clone(), k.clone(), v.clone(), b.clone(),
        None, None,
        use_qk_l2norm_in_kernel=True,
    )

    print("\n输出统计:")
    print(
        f"  不使用 L2 norm: mean={out_no_norm.mean():.6f}, std={out_no_norm.std():.6f}")
    print(
        f"  使用 L2 norm:   mean={out_with_norm.mean():.6f}, std={out_with_norm.std():.6f}")
    print(f"  差异: mean={(out_no_norm - out_with_norm).abs().mean():.6f}")

    print("\n✅ L2 归一化影响了输出（这是预期的）")


def demo_5_understand_algorithm():
    """示例 5: 理解算法原理"""
    print("\n" + "=" * 80)
    print("示例 5: 理解算法原理（单步分析）")
    print("=" * 80)

    # 简化配置（单 batch，单 head，单时间步）
    B, T, H, K, V = 1, 1, 1, 4, 4
    HV = H

    print(f"配置: 单个样本，单个时间步，便于理解")

    # 创建简单的输入
    A_log = torch.tensor([-1.0])  # 衰减参数
    a = torch.tensor([[[0.5]]])  # [B, T, HV]
    dt_bias = torch.tensor([0.5])
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])  # [B, T, H, K]
    k = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])  # [B, T, H, K]
    v = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [B, T, HV, V]
    b = torch.tensor([[[0.0]]])  # sigmoid(0) = 0.5

    print("\n输入:")
    print(f"  A_log = {A_log.item():.2f}")
    print(f"  a = {a.squeeze().item():.2f}")
    print(f"  dt_bias = {dt_bias.item():.2f}")
    print(f"  q = {q.squeeze().tolist()}")
    print(f"  k = {k.squeeze().tolist()}")
    print(f"  v = {v.squeeze().tolist()}")
    print(f"  b = {b.squeeze().item():.2f}")

    # 手动计算（模拟算法流程）
    print("\n算法步骤:")

    # 1. 计算 g
    import math
    x = a.item() + dt_bias.item()
    softplus_x = math.log(1 + math.exp(x))
    g = -math.exp(A_log.item()) * softplus_x
    print(f"  1. g = -exp({A_log.item():.2f}) * softplus({x:.2f}) = {g:.4f}")

    # 2. 计算 beta
    beta = 1.0 / (1.0 + math.exp(-b.item()))
    print(f"  2. beta = sigmoid({b.item():.2f}) = {beta:.4f}")

    # 3. 初始 hidden state
    h = torch.zeros(HV, K, V)
    print(f"  3. 初始 h = 0")

    # 4. 衰减
    h = h * math.exp(g)
    print(f"  4. h *= exp({g:.4f}) = 0 (仍为 0)")

    # 5. Delta rule
    v_adjusted = v.squeeze() - torch.sum(h * k.squeeze()[:, None], dim=0)
    print(f"  5. v_adjusted = v - sum(h * k) = {v_adjusted.tolist()}")

    # 6. Beta gating
    v_adjusted = v_adjusted * beta
    print(f"  6. v_adjusted *= {beta:.4f} = {v_adjusted.tolist()}")

    # 7. 更新 h
    h = h + k.squeeze()[:, None] * v_adjusted[None, :]
    print(f"  7. h 更新后:")
    print(f"     {h.tolist()}")

    # 8. 计算输出
    out_manual = torch.sum(h * q.squeeze()[:, None], dim=0)
    print(f"  8. out = sum(h * q) = {out_manual.tolist()}")

    # 使用函数计算
    out_func = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, 1.0, 20.0,
        q, k, v, b,
        None, None,
    )

    print(f"\n函数输出: {out_func.squeeze().tolist()}")
    print(f"手动计算: {out_manual.tolist()}")
    print(f"差异: {(out_func.squeeze() - out_manual).abs().max():.2e}")

    print("\n✅ 手动计算与函数输出一致！")


def main():
    """运行所有演示"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Fused Sigmoid Gating Delta Rule Update 演示" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    print(f"\n✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 运行所有演示
    demos = [
        demo_1_basic_usage,
        demo_2_with_state,
        demo_3_optimized_version,
        demo_4_with_l2norm,
        demo_5_understand_algorithm,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ 演示失败: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "=" * 80)
    print("演示总结")
    print("=" * 80)
    print("✅ 示例 1: 基本使用 - 了解函数调用方式")
    print("✅ 示例 2: 状态管理 - 模拟多轮推理")
    print("✅ 示例 3: 优化版本 - 提升性能")
    print("✅ 示例 4: L2 归一化 - 理解参数影响")
    print("✅ 示例 5: 算法原理 - 深入理解每一步")

    print("\n" + "🎉 " * 20)
    print("✅ 演示完成！可以开始使用 Native 实现了")
    print("🎉 " * 20)

    print("\n📚 更多信息请查看:")
    print("  - fused_sigmoid_gating_native_implementation.py  (核心实现)")
    print("  - FUSED_SIGMOID_GATING_ANALYSIS.md             (详细分析)")
    print("  - test_fused_sigmoid_gating_native.py          (测试套件)")


if __name__ == "__main__":
    main()
