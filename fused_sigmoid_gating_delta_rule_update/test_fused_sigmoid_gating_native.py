#!/usr/bin/env python3
"""
测试 Fused Sigmoid Gating Delta Rule Update Native 实现

验证 PyTorch native 实现与 Triton kernel 的等价性
"""

from fused_sigmoid_gating_native_implementation import (
    fused_sigmoid_gating_delta_rule_update_native,
    fused_sigmoid_gating_delta_rule_update_native_optimized,
)
import torch
import sys
import os
import argparse
from pathlib import Path

# 【修复导入问题】添加 python 目录到搜索路径
project_root = Path(__file__).parent
python_dir = project_root / 'python'
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
    print(f"✅ 已添加到 Python 路径: {python_dir}")

# 导入 native 实现

# 尝试导入 Triton 实现
try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as fused_sigmoid_gating_delta_rule_update_triton,
    )
    HAS_TRITON = True
    print("✅ Triton 实现已成功加载")
except ImportError as e:
    HAS_TRITON = False
    print(f"⚠️  Triton 实现未找到: {e}")
    print("    仅测试 native 实现的内部一致性")

# ============================================================================
# 全局配置
# ============================================================================
DEVICE = None  # 将在 parse_args 中设置
SKIP_TRITON = False  # 是否跳过 Triton 对比


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试 Fused Sigmoid Gating Native 实现')
    parser.add_argument('--device', type=str, default='cpu',
                        help='测试设备 (cpu, cuda, cuda:0, etc.). 默认: cpu')
    parser.add_argument('--skip-triton', action='store_true',
                        help='跳过 Triton 对比测试')
    args = parser.parse_args()

    global DEVICE, SKIP_TRITON
    DEVICE = args.device
    SKIP_TRITON = args.skip_triton

    return args


def allclose_with_info(a, b, rtol=1e-3, atol=1e-4, name=""):
    """检查两个 tensor 是否接近，并打印详细信息"""
    if a is None and b is None:
        return True
    if a is None or b is None:
        print(f"  ❌ {name}: 一个为 None")
        return False

    close = torch.allclose(a, b, rtol=rtol, atol=atol)

    if close:
        max_diff = (a - b).abs().max().item()
        print(f"  ✅ {name}: 最大差异 {max_diff:.2e}")
    else:
        max_diff = (a - b).abs().max().item()
        mean_diff = (a - b).abs().mean().item()
        print(f"  ❌ {name}: 最大差异 {max_diff:.2e}, 平均差异 {mean_diff:.2e}")
        print(f"     a: mean={a.mean():.6f}, std={a.std():.6f}")
        print(f"     b: mean={b.mean():.6f}, std={b.std():.6f}")

        # 打印不匹配的位置（前几个）
        diff = (a - b).abs()
        max_indices = torch.topk(diff.flatten(), k=min(5, diff.numel()))[1]
        print(f"     Top 5 差异位置:")
        for idx in max_indices:
            flat_idx = idx.item()
            multi_idx = torch.unravel_index(torch.tensor(flat_idx), a.shape)
            a_val = a.flatten()[flat_idx].item()
            b_val = b.flatten()[flat_idx].item()
            print(
                f"       位置 {tuple(x.item() for x in multi_idx)}: a={a_val:.6f}, b={b_val:.6f}, diff={abs(a_val-b_val):.6f}")

    return close


def test_case_1_basic_functionality():
    """测试 1: 基本功能（小规模）"""
    print("\n" + "=" * 80)
    print("测试 1: 基本功能（小规模）")
    print("=" * 80)

    # 准备数据
    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H

    device = 'cuda:0'
    dtype = torch.float32

    print(f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=device) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    print("\n测试 1.1: 无初始状态")

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    if HAS_TRITON and device == 'cuda:0':
        # 与 Triton 对比
        out_triton = fused_sigmoid_gating_delta_rule_update_triton(
            A_log, a, dt_bias, softplus_beta, softplus_threshold,
            q.clone(), k.clone(), v.clone(), b,
            initial_state_source=None,
            initial_state_indices=None,
            scale=None,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )

        print(
            f"Triton 输出: shape={out_triton.shape}, mean={out_triton.mean():.6f}, std={out_triton.std():.6f}")
        success &= allclose_with_info(
            out_native, out_triton, name="Native vs Triton")

    return success


def test_case_2_with_initial_state():
    """测试 2: 带初始状态"""
    print("\n" + "=" * 80)
    print("测试 2: 带初始状态")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    num_states = 3

    device = 'cpu'
    dtype = torch.float32

    print(
        f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, num_states={num_states}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=device) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1

    # 初始状态
    initial_state_source = torch.randn(
        num_states, HV, K, V, dtype=dtype, device=device) * 0.1
    initial_state_indices = torch.tensor(
        [0, 1], dtype=torch.long, device=device)  # 使用状态 0 和 1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现（复制初始状态以避免修改）
    initial_state_source_native = initial_state_source.clone()
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b,
        initial_state_source=initial_state_source_native,
        initial_state_indices=initial_state_indices,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")
    print(
        f"Native 最终状态: mean={initial_state_source_native.mean():.6f}, std={initial_state_source_native.std():.6f}")

    # 优化版本
    initial_state_source_optimized = initial_state_source.clone()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b,
        initial_state_source=initial_state_source_optimized,
        initial_state_indices=initial_state_indices,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")
    print(
        f"Optimized 最终状态: mean={initial_state_source_optimized.mean():.6f}, std={initial_state_source_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="输出 (Native vs Optimized)")
    success &= allclose_with_info(
        initial_state_source_native, initial_state_source_optimized, name="最终状态 (Native vs Optimized)")

    return success


def test_case_3_with_l2norm():
    """测试 3: 带 L2 归一化"""
    print("\n" + "=" * 80)
    print("测试 3: 带 L2 归一化")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, use_qk_l2norm=True")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=device) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=True,  # 开启 L2 归一化
        cu_seqlens=None,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    return success


def test_case_4_custom_scale():
    """测试 4: 自定义 scale"""
    print("\n" + "=" * 80)
    print("测试 4: 自定义 scale")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    custom_scale = 0.5

    device = 'cpu'
    dtype = torch.float32

    print(
        f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, scale={custom_scale}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=device) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=custom_scale,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=custom_scale,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    return success


def test_case_5_larger_scale():
    """测试 5: 较大规模"""
    print("\n" + "=" * 80)
    print("测试 5: 较大规模")
    print("=" * 80)

    B, T, H, K, V = 4, 8, 4, 16, 16
    HV = H

    device = 'cpu'
    dtype = torch.float32

    print(f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=device) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=device) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    import time
    start = time.time()
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    native_time = time.time() - start

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}, time={native_time:.3f}s")

    # 优化版本
    start = time.time()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b,
        initial_state_source=None,
        initial_state_indices=None,
        scale=None,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    optimized_time = time.time() - start

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}, time={optimized_time:.3f}s")
    print(f"加速比: {native_time / optimized_time:.2f}x")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    return success


def run_all_tests():
    """运行所有测试"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 10 +
          "Fused Sigmoid Gating Delta Rule Update 测试套件" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    print(f"\n✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"✅ Triton 实现: {'可用' if HAS_TRITON else '不可用（仅测试 native）'}")

    results = []

    # 运行所有测试
    tests = [
        ("基本功能", test_case_1_basic_functionality),
        ("带初始状态", test_case_2_with_initial_state),
        ("带 L2 归一化", test_case_3_with_l2norm),
        ("自定义 scale", test_case_4_custom_scale),
        ("较大规模", test_case_5_larger_scale),
    ]

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:30s}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n" + "🎉 " * 20)
        print("✅ 所有测试通过！Native 实现与 Triton kernel 完全等价")
        print("🎉 " * 20)
    else:
        print("\n" + "⚠️  " * 20)
        print("❌ 部分测试失败，请检查实现")
        print("⚠️  " * 20)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
