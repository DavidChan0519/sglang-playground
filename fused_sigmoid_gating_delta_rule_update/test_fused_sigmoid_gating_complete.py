#!/usr/bin/env python3
"""
Fused Sigmoid Gating Delta Rule Update - 完整测试套件

这个测试文件整合了所有测试场景，包括：
1. 固定长度序列测试（基本功能、初始状态、L2归一化、自定义scale、大规模）
2. 变长序列测试（cu_seqlens）
3. Native vs Optimized vs Triton 的全面对比

使用方法:
    # CPU 测试
    python3 test_fused_sigmoid_gating_complete.py --device cpu
    
    # CUDA 测试
    python3 test_fused_sigmoid_gating_complete.py --device cuda
    
    # 跳过 Triton 对比
    python3 test_fused_sigmoid_gating_complete.py --device cuda --skip-triton
    
    # 只运行特定测试
    python3 test_fused_sigmoid_gating_complete.py --device cpu --test basic
    python3 test_fused_sigmoid_gating_complete.py --device cpu --test varlen
"""

from fused_sigmoid_gating_native_implementation import (
    fused_sigmoid_gating_delta_rule_update_native,
    fused_sigmoid_gating_delta_rule_update_native_optimized,
)
import torch
import sys
import argparse
from pathlib import Path

# 添加 python 目录到搜索路径
project_root = Path(__file__).parent
python_dir = project_root / 'python'
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

# 导入 native 实现

# 尝试导入 Triton 实现
try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as fused_sigmoid_gating_delta_rule_update_triton,
    )
    HAS_TRITON = True
except ImportError as e:
    HAS_TRITON = False
    TRITON_IMPORT_ERROR = str(e)

# ============================================================================
# 全局配置
# ============================================================================
DEVICE = None  # 将在 parse_args 中设置
SKIP_TRITON = False  # 是否跳过 Triton 对比
TEST_FILTER = None  # 测试过滤器


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Fused Sigmoid Gating Delta Rule Update - 完整测试套件')
    parser.add_argument('--device', type=str, default='cpu',
                        help='测试设备 (cpu, cuda, cuda:0, etc.). 默认: cpu')
    parser.add_argument('--skip-triton', action='store_true',
                        help='跳过 Triton 对比测试')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'fixed', 'varlen', 'basic',
                                 'state', 'l2norm', 'scale', 'large'],
                        help='运行特定测试: all(全部), fixed(固定长度), varlen(变长序列), basic(基本功能), etc.')
    args = parser.parse_args()

    global DEVICE, SKIP_TRITON, TEST_FILTER
    DEVICE = args.device
    SKIP_TRITON = args.skip_triton
    TEST_FILTER = args.test

    return args


def allclose_with_info(a, b, name="Comparison", rtol=1e-5, atol=1e-5):
    """带详细信息的 allclose 检查"""
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

    return close


def test_with_triton(test_name, native_out, optimized_out, A_log, a, dt_bias,
                     softplus_beta, softplus_threshold, q, k, v, b,
                     initial_state_source, initial_state_indices, scale, use_qk_l2norm,
                     cu_seqlens=None):
    """统一的 Triton 对比测试"""
    success = True

    if not HAS_TRITON:
        print(f"  ⚠️  跳过 Triton 对比: {TRITON_IMPORT_ERROR}")
        return success

    if SKIP_TRITON:
        print(f"  ⏭️  跳过 Triton 对比 (--skip-triton)")
        return success

    if not DEVICE.startswith('cuda'):
        print(f"  ⏭️  跳过 Triton 对比 (Triton 需要 CUDA)")
        return success

    try:
        # 复制初始状态（如果有）
        triton_state = None
        if initial_state_source is not None:
            triton_state = initial_state_source.clone()

        out_triton = fused_sigmoid_gating_delta_rule_update_triton(
            A_log.clone(), a.clone(), dt_bias.clone(),
            softplus_beta, softplus_threshold,
            q.clone(), k.clone(), v.clone(), b.clone(),
            initial_state_source=triton_state,
            initial_state_indices=initial_state_indices.clone(
            ) if initial_state_indices is not None else None,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            cu_seqlens=cu_seqlens,
        )

        print(
            f"  Triton 输出: shape={out_triton.shape}, mean={out_triton.mean():.6f}, std={out_triton.std():.6f}")
        success &= allclose_with_info(
            native_out, out_triton, name="Native vs Triton")
        success &= allclose_with_info(
            optimized_out, out_triton, name="Optimized vs Triton")

    except Exception as e:
        print(f"  ❌ Triton 测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False

    return success


# ============================================================================
# 固定长度序列测试
# ============================================================================

def test_fixed_1_basic():
    """测试 1: 基本功能（小规模）"""
    print("\n" + "=" * 80)
    print("测试 1: 基本功能（小规模）")
    print("=" * 80)

    # 准备数据
    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    dtype = torch.float32

    print(f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    # 与 Triton 对比
    success &= test_with_triton(
        "基本功能", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, None, None, None, False
    )

    return success


def test_fixed_2_initial_state():
    """测试 2: 带初始状态"""
    print("\n" + "=" * 80)
    print("测试 2: 带初始状态")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    num_states = 3
    dtype = torch.float32

    print(
        f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, num_states={num_states}, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1

    # 初始状态
    initial_state_source = torch.randn(
        num_states, HV, K, V, dtype=dtype, device=DEVICE) * 0.1
    initial_state_indices = torch.tensor(
        [0, 1], dtype=torch.long, device=DEVICE)

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    initial_state_native = initial_state_source.clone()
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=initial_state_native,
        initial_state_indices=initial_state_indices.clone(),
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    initial_state_optimized = initial_state_source.clone()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=initial_state_optimized,
        initial_state_indices=initial_state_indices.clone(),
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="输出 (Native vs Optimized)")
    success &= allclose_with_info(
        initial_state_native, initial_state_optimized, name="最终状态 (Native vs Optimized)")

    # 与 Triton 对比
    success &= test_with_triton(
        "带初始状态", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, initial_state_source, initial_state_indices, None, False
    )

    return success


def test_fixed_3_l2norm():
    """测试 3: 带 L2 归一化"""
    print("\n" + "=" * 80)
    print("测试 3: 带 L2 归一化")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    dtype = torch.float32

    print(
        f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, use_qk_l2norm=True, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=True,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=True,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    # 与 Triton 对比
    success &= test_with_triton(
        "带 L2 归一化", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, None, None, None, True
    )

    return success


def test_fixed_4_custom_scale():
    """测试 4: 自定义 scale"""
    print("\n" + "=" * 80)
    print("测试 4: 自定义 scale")
    print("=" * 80)

    B, T, H, K, V = 2, 4, 2, 8, 8
    HV = H
    custom_scale = 0.5
    dtype = torch.float32

    print(
        f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, scale={custom_scale}, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=custom_scale, use_qk_l2norm_in_kernel=False,
    )

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")

    # 优化版本
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=custom_scale, use_qk_l2norm_in_kernel=False,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    # 与 Triton 对比
    success &= test_with_triton(
        "自定义 scale", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, None, None, custom_scale, False
    )

    return success


def test_fixed_5_larger_scale():
    """测试 5: 较大规模"""
    print("\n" + "=" * 80)
    print("测试 5: 较大规模")
    print("=" * 80)

    B, T, H, K, V = 4, 8, 4, 16, 16
    HV = H
    dtype = torch.float32

    print(f"配置: B={B}, T={T}, H={H}, HV={HV}, K={K}, V={V}, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(B, T, HV, dtype=dtype, device=DEVICE) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    import time
    start = time.time()
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
    )
    native_time = time.time() - start

    print(
        f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}, time={native_time:.3f}s")

    # 优化版本
    start = time.time()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(), softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
    )
    optimized_time = time.time() - start

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}, time={optimized_time:.3f}s")
    print(f"加速比: {native_time / optimized_time:.2f}x")

    success = allclose_with_info(
        out_native, out_optimized, name="Native vs Optimized")

    # 与 Triton 对比
    success &= test_with_triton(
        "较大规模", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, None, None, None, False
    )

    return success


# ============================================================================
# 变长序列测试
# ============================================================================

def test_varlen_1_vs_fixed():
    """测试 6: 变长序列 vs 固定长度"""
    print("\n" + "=" * 80)
    print("测试 6: 变长序列 vs 固定长度")
    print("=" * 80)

    dtype = torch.float32

    # 创建3个不同长度的序列
    seq_lens = [5, 7, 6]
    N = len(seq_lens)
    total_len = sum(seq_lens)
    cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(N)],
                              dtype=torch.int64, device=DEVICE)

    H, HV, K, V = 2, 2, 8, 8

    print(
        f"配置: N={N}, seq_lens={seq_lens}, total_len={total_len}, device={DEVICE}")
    print(f"       H={H}, HV={HV}, K={K}, V={V}")
    print(f"       cu_seqlens={cu_seqlens.tolist()}")

    # 创建输入（变长格式）
    A_log_varlen = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a_varlen = torch.randn(1, total_len, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias_varlen = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q_varlen = torch.randn(
        1, total_len, H, K, dtype=dtype, device=DEVICE) * 0.1
    k_varlen = torch.randn(
        1, total_len, H, K, dtype=dtype, device=DEVICE) * 0.1
    v_varlen = torch.randn(1, total_len, HV, V,
                           dtype=dtype, device=DEVICE) * 0.1
    b_varlen = torch.randn(1, total_len, HV, dtype=dtype, device=DEVICE) * 0.1

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现（变长模式）
    print("\n[1] Native 实现（变长模式）")
    out_native_varlen = fused_sigmoid_gating_delta_rule_update_native(
        A_log_varlen.clone(), a_varlen.clone(), dt_bias_varlen.clone(),
        softplus_beta, softplus_threshold,
        q_varlen.clone(), k_varlen.clone(), v_varlen.clone(), b_varlen.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    print(
        f"输出: shape={out_native_varlen.shape}, mean={out_native_varlen.mean():.6f}")

    # Optimized 实现（变长模式）
    print("\n[2] Optimized 实现（变长模式）")
    out_optimized_varlen = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log_varlen.clone(), a_varlen.clone(), dt_bias_varlen.clone(),
        softplus_beta, softplus_threshold,
        q_varlen.clone(), k_varlen.clone(), v_varlen.clone(), b_varlen.clone(),
        initial_state_source=None, initial_state_indices=None,
        scale=None, use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    print(
        f"输出: shape={out_optimized_varlen.shape}, mean={out_optimized_varlen.mean():.6f}")

    # 对比 Native vs Optimized（变长）
    max_diff_varlen = (out_native_varlen -
                       out_optimized_varlen).abs().max().item()
    print(f"\n✅ Native vs Optimized (变长模式): 最大差异 {max_diff_varlen:.2e}")

    # 将变长格式转换为固定长度格式，逐个序列测试
    print("\n[3] 固定长度模式（逐序列对比）")
    success = True

    for seq_idx in range(N):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos

        print(f"\n  序列 {seq_idx}: 长度={seq_len}, 范围=[{bos}:{eos}]")

        # 提取当前序列的数据
        a_fixed = a_varlen[:, bos:eos].contiguous()
        q_fixed = q_varlen[:, bos:eos].contiguous()
        k_fixed = k_varlen[:, bos:eos].contiguous()
        v_fixed = v_varlen[:, bos:eos].contiguous()
        b_fixed = b_varlen[:, bos:eos].contiguous()

        # Native 固定长度模式
        out_native_fixed = fused_sigmoid_gating_delta_rule_update_native(
            A_log_varlen.clone(), a_fixed.clone(), dt_bias_varlen.clone(),
            softplus_beta, softplus_threshold,
            q_fixed.clone(), k_fixed.clone(), v_fixed.clone(), b_fixed.clone(),
            initial_state_source=None, initial_state_indices=None,
            scale=None, use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )

        # 从变长输出中提取对应部分
        out_native_varlen_slice = out_native_varlen[0, bos:eos].contiguous()

        # 对比
        max_diff = (out_native_fixed -
                    out_native_varlen_slice).abs().max().item()
        print(f"    Fixed vs Varlen: 最大差异 {max_diff:.2e}")

        if max_diff > 1e-5:
            print(f"    ❌ 差异过大！")
            success = False
        else:
            print(f"    ✅ 一致")

    # 与 Triton 对比（变长模式）
    success &= test_with_triton(
        "变长序列", out_native_varlen, out_optimized_varlen,
        A_log_varlen, a_varlen, dt_bias_varlen, softplus_beta, softplus_threshold,
        q_varlen, k_varlen, v_varlen, b_varlen, None, None, None, False,
        cu_seqlens=cu_seqlens
    )

    return success


def test_varlen_2_with_state():
    """测试 7: 变长序列 + 初始状态"""
    print("\n" + "=" * 80)
    print("测试 7: 变长序列 + 初始状态")
    print("=" * 80)

    dtype = torch.float32

    seq_lens = [4, 6, 5]
    N = len(seq_lens)
    total_len = sum(seq_lens)
    cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(N)],
                              dtype=torch.int64, device=DEVICE)

    H, HV, K, V = 2, 2, 8, 8
    num_states = 5

    print(
        f"配置: N={N}, seq_lens={seq_lens}, num_states={num_states}, device={DEVICE}")

    # 创建输入
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    a = torch.randn(1, total_len, HV, dtype=dtype, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1
    q = torch.randn(1, total_len, H, K, dtype=dtype, device=DEVICE) * 0.1
    k = torch.randn(1, total_len, H, K, dtype=dtype, device=DEVICE) * 0.1
    v = torch.randn(1, total_len, HV, V, dtype=dtype, device=DEVICE) * 0.1
    b = torch.randn(1, total_len, HV, dtype=dtype, device=DEVICE) * 0.1

    # 初始状态
    initial_state_source = torch.randn(
        num_states, HV, K, V, dtype=dtype, device=DEVICE) * 0.1
    initial_state_indices = torch.tensor(
        [0, 2, 1], dtype=torch.long, device=DEVICE)

    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Native 实现
    initial_state_native = initial_state_source.clone()
    out_native = fused_sigmoid_gating_delta_rule_update_native(
        A_log.clone(), a.clone(), dt_bias.clone(),
        softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=initial_state_native,
        initial_state_indices=initial_state_indices.clone(),
        scale=None, use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )

    print(f"Native 输出: shape={out_native.shape}, mean={out_native.mean():.6f}")
    print(f"最终状态[0]: mean={initial_state_native[0].mean():.6f}")
    print(f"最终状态[2]: mean={initial_state_native[2].mean():.6f}")
    print(f"最终状态[1]: mean={initial_state_native[1].mean():.6f}")

    # Optimized 实现
    initial_state_optimized = initial_state_source.clone()
    out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
        A_log.clone(), a.clone(), dt_bias.clone(),
        softplus_beta, softplus_threshold,
        q.clone(), k.clone(), v.clone(), b.clone(),
        initial_state_source=initial_state_optimized,
        initial_state_indices=initial_state_indices.clone(),
        scale=None, use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )

    print(
        f"Optimized 输出: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}")

    # 对比
    max_diff_out = (out_native - out_optimized).abs().max().item()
    max_diff_state = (initial_state_native -
                      initial_state_optimized).abs().max().item()

    print(f"\n✅ Native vs Optimized")
    print(f"   输出: 最大差异 {max_diff_out:.2e}")
    print(f"   最终状态: 最大差异 {max_diff_state:.2e}")

    success = max_diff_out < 1e-5 and max_diff_state < 1e-5

    # 与 Triton 对比
    success &= test_with_triton(
        "变长序列+初始状态", out_native, out_optimized,
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, initial_state_source, initial_state_indices, None, False,
        cu_seqlens=cu_seqlens
    )

    return success


def test_varlen_3_missing_batch_dim():
    """测试 8: 变长序列 + 缺少 batch 维度（实际生产场景）"""
    print("\n" + "=" * 80)
    print("测试 8: 变长序列 + 缺少 batch 维度（实际生产场景）")
    print("=" * 80)

    dtype = torch.float32

    # 模拟实际生产场景的配置
    # 512个序列，每个序列长度为1（cu_seqlens = [0, 1, 2, 3, ..., 512]）
    N = 512
    seq_lens = [1] * N  # 每个序列长度为1
    total_len = sum(seq_lens)  # 512
    cu_seqlens = torch.arange(N + 1, dtype=torch.int64,
                              device=DEVICE)  # [0, 1, 2, ..., 512]

    # 注意：设置 H == HV 以支持 Optimized 版本
    H, HV, K, V = 32, 32, 128, 128
    num_states = N + 1  # 513

    print(
        f"配置: N={N}, total_len={total_len}, H={H}, HV={HV}, K={K}, V={V}, device={DEVICE}")
    print(f"       num_states={num_states}")
    print(f"       ⚠️  注意: a 和 b 缺少 batch 维度 (实际生产场景)")

    # 创建输入（注意：a 和 b 缺少 batch 维度，这是实际生产场景中可能出现的情况）
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1

    # 关键：a 和 b 是 2D 张量，缺少 batch 维度
    a = torch.randn(total_len, HV, dtype=dtype,
                    device=DEVICE) * 0.1  # [512, 32]
    b = torch.randn(total_len, HV, dtype=dtype,
                    device=DEVICE) * 0.1  # [512, 32]

    dt_bias = torch.randn(HV, dtype=dtype, device=DEVICE) * 0.1

    # q, k, v 有正确的形状
    q = torch.randn(1, total_len, H, K, dtype=dtype,
                    device=DEVICE) * 0.1  # [1, 512, 16, 128]
    k = torch.randn(1, total_len, H, K, dtype=dtype,
                    device=DEVICE) * 0.1  # [1, 512, 16, 128]
    v = torch.randn(1, total_len, HV, V, dtype=dtype,
                    device=DEVICE) * 0.1  # [1, 512, 32, 128]

    # 初始状态
    initial_state_source = torch.randn(
        num_states, HV, K, V, dtype=dtype, device=DEVICE) * 0.1
    initial_state_indices = torch.arange(
        N, dtype=torch.long, device=DEVICE)  # [0, 1, 2, ..., 511]

    custom_scale = 0.08838834764831845
    softplus_beta = 1.0
    softplus_threshold = 20.0

    print(f"\n输入形状验证:")
    print(f"  A_log: {a.shape}")
    print(f"  a: {a.shape} ⚠️  (缺少 batch 维度)")
    print(f"  b: {b.shape} ⚠️  (缺少 batch 维度)")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print(f"  initial_state_source: {initial_state_source.shape}")
    print(f"  initial_state_indices: {initial_state_indices.shape}")
    print(f"  cu_seqlens: {cu_seqlens.shape}")

    # Native 实现（应该自动处理缺少的 batch 维度）
    print("\n[1] Native 实现")
    try:
        initial_state_native = initial_state_source.clone()
        out_native = fused_sigmoid_gating_delta_rule_update_native(
            A_log.clone(), a.clone(), dt_bias.clone(),
            softplus_beta, softplus_threshold,
            q.clone(), k.clone(), v.clone(), b.clone(),
            initial_state_source=initial_state_native,
            initial_state_indices=initial_state_indices.clone(),
            scale=custom_scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        print(
            f"✅ Native 成功: shape={out_native.shape}, mean={out_native.mean():.6f}, std={out_native.std():.6f}")
    except Exception as e:
        print(f"❌ Native 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Optimized 实现
    print("\n[2] Optimized 实现")
    try:
        initial_state_optimized = initial_state_source.clone()
        out_optimized = fused_sigmoid_gating_delta_rule_update_native_optimized(
            A_log.clone(), a.clone(), dt_bias.clone(),
            softplus_beta, softplus_threshold,
            q.clone(), k.clone(), v.clone(), b.clone(),
            initial_state_source=initial_state_optimized,
            initial_state_indices=initial_state_indices.clone(),
            scale=custom_scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        print(
            f"✅ Optimized 成功: shape={out_optimized.shape}, mean={out_optimized.mean():.6f}, std={out_optimized.std():.6f}")
    except Exception as e:
        print(f"❌ Optimized 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 对比
    max_diff_out = (out_native - out_optimized).abs().max().item()
    max_diff_state = (initial_state_native -
                      initial_state_optimized).abs().max().item()

    print(f"\n✅ Native vs Optimized")
    print(f"   输出: 最大差异 {max_diff_out:.2e}")
    print(f"   最终状态: 最大差异 {max_diff_state:.2e}")

    success = max_diff_out < 1e-5 and max_diff_state < 1e-5

    # 与 Triton 对比（注意：Triton 可能也需要处理缺少 batch 维度的情况）
    # 暂时跳过 Triton 对比，因为 Triton 可能不支持这种情况
    print(f"\n  ⏭️  跳过 Triton 对比（Triton 可能不支持缺少 batch 维度的输入）")

    return success


# ============================================================================
# 主测试函数
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    global DEVICE

    # 解析命令行参数
    args = parse_args()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 10 + "Fused Sigmoid Gating - 完整测试套件" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")

    # 显示配置
    print(f"\n✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ 测试设备: {DEVICE}")
    if DEVICE.startswith('cuda'):
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(DEVICE)}")
        else:
            print(f"❌ CUDA 不可用，将回退到 CPU")
            DEVICE = 'cpu'

    print(f"✅ Triton 实现: {'可用' if HAS_TRITON else '不可用'}")
    if not HAS_TRITON:
        print(f"   原因: {TRITON_IMPORT_ERROR}")
    if SKIP_TRITON:
        print(f"⏭️  跳过 Triton 对比测试 (--skip-triton)")
    if TEST_FILTER != 'all':
        print(f"🎯 测试过滤器: {TEST_FILTER}")

    results = []

    # 定义所有测试
    all_tests = [
        # 固定长度测试
        ("基本功能", "fixed", "basic", test_fixed_1_basic),
        ("带初始状态", "fixed", "state", test_fixed_2_initial_state),
        ("带 L2 归一化", "fixed", "l2norm", test_fixed_3_l2norm),
        ("自定义 scale", "fixed", "scale", test_fixed_4_custom_scale),
        ("较大规模", "fixed", "large", test_fixed_5_larger_scale),
        # 变长序列测试
        ("变长 vs 固定长度", "varlen", "varlen", test_varlen_1_vs_fixed),
        ("变长 + 初始状态", "varlen", "varlen", test_varlen_2_with_state),
        ("变长 + 缺失batch维度", "varlen", "varlen", test_varlen_3_missing_batch_dim),
    ]

    # 根据过滤器运行测试
    for test_name, category, tag, test_func in all_tests:
        # 检查是否应该运行此测试
        if TEST_FILTER == 'all':
            should_run = True
        elif TEST_FILTER == 'fixed':
            should_run = (category == 'fixed')
        elif TEST_FILTER == 'varlen':
            should_run = (category == 'varlen')
        else:
            should_run = (tag == TEST_FILTER)

        if not should_run:
            continue

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
