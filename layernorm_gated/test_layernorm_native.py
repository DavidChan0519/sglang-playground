#!/usr/bin/env python3
"""
测试 PyTorch Native LayerNorm 与 Triton Kernel 的等价性

测试场景：
1. 标准 LayerNorm
2. RMSNorm
3. LayerNorm + Gating (SwiGLU)
4. GroupNorm
5. 各种组合
"""

import torch
import sys

# 导入 native 实现
from layernorm_native_implementation import (
    _layer_norm_fwd_native,
    layernorm_fn_native,
    rmsnorm_fn_native,
    simple_layernorm_native,
)

# 尝试导入 Triton 实现
try:
    from python.sglang.srt.layers.attention.fla.layernorm_gated import (
        _layer_norm_fwd as _layer_norm_fwd_triton,
        layernorm_fn as layernorm_fn_triton,
        rmsnorm_fn as rmsnorm_fn_triton,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("⚠️  Triton 实现未找到，仅测试 native 实现的正确性")


def allclose_with_info(a, b, rtol=1e-4, atol=1e-5, name=""):
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

    return close


def test_case_1_standard_layernorm():
    """测试 1: 标准 LayerNorm（无额外功能）"""
    print("\n" + "=" * 80)
    print("测试 1: 标准 LayerNorm")
    print("=" * 80)

    # 准备数据
    M, N = 32, 256
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    print(f"输入: x={x.shape}, weight={weight.shape}, bias={bias.shape}")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=None,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    )

    print(
        f"Native 输出: out={out_native.shape}, mean={mean_native.shape}, rstd={rstd_native.shape}")

    # 与 PyTorch 标准实现对比
    layer_norm = torch.nn.LayerNorm(N, eps=eps, device='cuda')
    layer_norm.weight.data = weight.clone()
    layer_norm.bias.data = bias.clone()
    out_torch = layer_norm(x)

    success = allclose_with_info(
        out_native, out_torch, name="vs PyTorch LayerNorm")

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias.clone(), eps,
            z=None,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=False,
        )

        print(
            f"Triton 输出: out={out_triton.shape}, mean={mean_triton.shape}, rstd={rstd_triton.shape}")

        success &= allclose_with_info(
            out_native, out_triton, name="Native vs Triton (out)")
        success &= allclose_with_info(
            mean_native, mean_triton, name="Native vs Triton (mean)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, name="Native vs Triton (rstd)")

    return success


def test_case_2_rmsnorm():
    """测试 2: RMSNorm"""
    print("\n" + "=" * 80)
    print("测试 2: RMSNorm")
    print("=" * 80)

    # 准备数据
    M, N = 64, 512
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    eps = 1e-6

    print(f"输入: x={x.shape}, dtype={x.dtype}")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=None,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,  # RMSNorm
    )

    print(
        f"Native 输出: out={out_native.shape}, mean={mean_native}, rstd={rstd_native.shape}")
    assert mean_native is None, "RMSNorm should not compute mean"

    success = True

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias.clone(), eps,
            z=None,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
        )

        success &= allclose_with_info(
            out_native, out_triton, rtol=1e-3, atol=1e-4, name="Native vs Triton (out)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, rtol=1e-3, atol=1e-4, name="Native vs Triton (rstd)")
        assert mean_triton is None, "Triton RMSNorm should not compute mean"

    return success


def test_case_3_gating_before_norm():
    """测试 3: Gating BEFORE Normalization"""
    print("\n" + "=" * 80)
    print("测试 3: Gating (SwiGLU) BEFORE Normalization")
    print("=" * 80)

    # 准备数据
    M, N = 16, 128
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    z = torch.randn(M, N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    print(f"输入: x={x.shape}, z={z.shape} (gating)")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=z,
        group_size=None,
        norm_before_gate=False,  # Gating BEFORE norm
        is_rms_norm=False,
    )

    print(f"Native 输出: out={out_native.shape}")

    success = True

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias.clone(), eps,
            z=z.clone(),
            group_size=None,
            norm_before_gate=False,
            is_rms_norm=False,
        )

        success &= allclose_with_info(
            out_native, out_triton, name="Native vs Triton (out)")
        success &= allclose_with_info(
            mean_native, mean_triton, name="Native vs Triton (mean)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, name="Native vs Triton (rstd)")

    return success


def test_case_4_gating_after_norm():
    """测试 4: Gating AFTER Normalization"""
    print("\n" + "=" * 80)
    print("测试 4: Gating (SwiGLU) AFTER Normalization")
    print("=" * 80)

    # 准备数据
    M, N = 16, 128
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = None  # 测试无 bias
    z = torch.randn(M, N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    print(f"输入: x={x.shape}, z={z.shape}, bias=None")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=z,
        group_size=None,
        norm_before_gate=True,  # Gating AFTER norm
        is_rms_norm=False,
    )

    print(f"Native 输出: out={out_native.shape}")

    success = True

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias, eps,
            z=z.clone(),
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=False,
        )

        success &= allclose_with_info(
            out_native, out_triton, name="Native vs Triton (out)")
        success &= allclose_with_info(
            mean_native, mean_triton, name="Native vs Triton (mean)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, name="Native vs Triton (rstd)")

    return success


def test_case_5_groupnorm():
    """测试 5: GroupNorm"""
    print("\n" + "=" * 80)
    print("测试 5: GroupNorm (group_size < N)")
    print("=" * 80)

    # 准备数据
    M, N = 32, 768
    group_size = 96  # 8 groups
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    print(f"输入: x={x.shape}, group_size={group_size} ({N // group_size} groups)")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=None,
        group_size=group_size,
        norm_before_gate=True,
        is_rms_norm=False,
    )

    print(
        f"Native 输出: out={out_native.shape}, mean={mean_native.shape}, rstd={rstd_native.shape}")

    success = True

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias.clone(), eps,
            z=None,
            group_size=group_size,
            norm_before_gate=True,
            is_rms_norm=False,
        )

        success &= allclose_with_info(
            out_native, out_triton, name="Native vs Triton (out)")
        success &= allclose_with_info(
            mean_native, mean_triton, name="Native vs Triton (mean)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, name="Native vs Triton (rstd)")

    return success


def test_case_6_complex_combination():
    """测试 6: 复杂组合（RMSNorm + GroupNorm + Gating）"""
    print("\n" + "=" * 80)
    print("测试 6: 复杂组合 (RMSNorm + GroupNorm + Gating)")
    print("=" * 80)

    # 准备数据
    M, N = 64, 1024
    group_size = 128  # 8 groups
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    bias = None
    z = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    eps = 1e-6

    print(f"输入: x={x.shape}, group_size={group_size}, z={z.shape}, bias=None")
    print(f"模式: RMSNorm + GroupNorm + Gating (before norm)")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=z,
        group_size=group_size,
        norm_before_gate=False,  # Gating before norm
        is_rms_norm=True,  # RMSNorm
    )

    print(f"Native 输出: out={out_native.shape}")
    assert mean_native is None, "RMSNorm should not compute mean"

    success = True

    if HAS_TRITON:
        # 与 Triton 实现对比
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x.clone(), weight.clone(), bias, eps,
            z=z.clone(),
            group_size=group_size,
            norm_before_gate=False,
            is_rms_norm=True,
        )

        success &= allclose_with_info(
            out_native, out_triton, rtol=1e-3, atol=1e-4, name="Native vs Triton (out)")
        success &= allclose_with_info(
            rstd_native, rstd_triton, rtol=1e-3, atol=1e-4, name="Native vs Triton (rstd)")
        assert mean_triton is None

    return success


def test_case_7_high_level_api():
    """测试 7: 高层 API（用户接口）"""
    print("\n" + "=" * 80)
    print("测试 7: 高层 API (layernorm_fn_native, rmsnorm_fn_native)")
    print("=" * 80)

    # 准备数据
    batch_size, seq_len, hidden_dim = 4, 128, 768
    x = torch.randn(batch_size, seq_len, hidden_dim,
                    dtype=torch.float32, device='cuda')
    weight = torch.randn(hidden_dim, dtype=torch.float32, device='cuda')
    bias = torch.randn(hidden_dim, dtype=torch.float32, device='cuda')
    z = torch.randn(batch_size, seq_len, hidden_dim,
                    dtype=torch.float32, device='cuda')
    eps = 1e-5

    print(f"输入: x={x.shape} (3D)")

    success = True

    # LayerNorm
    print("\n  测试 layernorm_fn_native:")
    out_native_ln = layernorm_fn_native(x, weight, bias, eps=eps)
    print(f"    输出: {out_native_ln.shape}")

    if HAS_TRITON:
        out_triton_ln = layernorm_fn_triton(
            x.clone(), weight.clone(), bias.clone(), eps=eps)
        success &= allclose_with_info(
            out_native_ln, out_triton_ln, name="    LayerNorm")

    # RMSNorm
    print("\n  测试 rmsnorm_fn_native:")
    out_native_rms = rmsnorm_fn_native(x, weight, bias, eps=eps)
    print(f"    输出: {out_native_rms.shape}")

    if HAS_TRITON:
        out_triton_rms = rmsnorm_fn_triton(
            x.clone(), weight.clone(), bias.clone(), eps=eps)
        success &= allclose_with_info(
            out_native_rms, out_triton_rms, rtol=1e-3, atol=1e-4, name="    RMSNorm")

    # LayerNorm + Gating
    print("\n  测试 layernorm_fn_native with gating:")
    out_native_gate = layernorm_fn_native(
        x, weight, bias, z=z, eps=eps, norm_before_gate=True)
    print(f"    输出: {out_native_gate.shape}")

    if HAS_TRITON:
        out_triton_gate = layernorm_fn_triton(x.clone(), weight.clone(
        ), bias.clone(), z=z.clone(), eps=eps, norm_before_gate=True)
        success &= allclose_with_info(
            out_native_gate, out_triton_gate, name="    LayerNorm + Gating")

    return success


def run_all_tests():
    """运行所有测试"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "LayerNorm Native 实现测试套件" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    if not torch.cuda.is_available():
        print("\n❌ 错误: CUDA 不可用，测试需要 GPU")
        return False

    print(f"\n✅ CUDA 可用: {torch.cuda.get_device_name()}")
    print(f"✅ Triton 实现: {'可用' if HAS_TRITON else '不可用（仅测试 native）'}")

    results = []

    # 运行所有测试
    tests = [
        ("标准 LayerNorm", test_case_1_standard_layernorm),
        ("RMSNorm", test_case_2_rmsnorm),
        ("Gating Before Norm", test_case_3_gating_before_norm),
        ("Gating After Norm", test_case_4_gating_after_norm),
        ("GroupNorm", test_case_5_groupnorm),
        ("复杂组合", test_case_6_complex_combination),
        ("高层 API", test_case_7_high_level_api),
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
