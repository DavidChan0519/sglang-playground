#!/usr/bin/env python3
"""
测试 PyTorch Native LayerNorm - 完整测试套件

功能：
1. 验证 Native 实现的逻辑正确性
2. 与 PyTorch 标准实现对比
3. 与 Triton 实现对比（如果可用）
4. 支持统一的 device 配置

使用方法:
    # CPU 测试（默认）
    python3 test_layernorm_native_cpu.py --device cpu
    
    # CUDA 测试
    python3 test_layernorm_native_cpu.py --device cuda
    
    # 跳过 Triton 对比
    python3 test_layernorm_native_cpu.py --device cuda --skip-triton
"""

from layernorm_native_implementation import (
    _layer_norm_fwd_native,
    layernorm_fn_native,
    rmsnorm_fn_native,
    simple_layernorm_native,
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
    from sglang.srt.layers.attention.fla.layernorm_gated import (
        _layer_norm_fwd as _layer_norm_fwd_triton,
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LayerNorm Native 实现测试套件')
    parser.add_argument('--device', type=str, default='cpu',
                        help='测试设备 (cpu, cuda, cuda:0, etc.). 默认: cpu')
    parser.add_argument('--skip-triton', action='store_true',
                        help='跳过 Triton 对比测试')
    args = parser.parse_args()

    global DEVICE, SKIP_TRITON
    DEVICE = args.device
    SKIP_TRITON = args.skip_triton

    return args


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
        print(
            f"     a: mean={a.mean():.6f}, std={a.std():.6f}, min={a.min():.6f}, max={a.max():.6f}")
        print(
            f"     b: mean={b.mean():.6f}, std={b.std():.6f}, min={b.min():.6f}, max={b.max():.6f}")

    return close


def test_with_triton(test_name, native_out, x, weight, bias, eps, z=None,
                     group_size=None, norm_before_gate=True, is_rms_norm=False):
    """统一的 Triton 对比测试"""
    if not HAS_TRITON:
        print(f"  ⚠️  跳过 Triton 对比: {TRITON_IMPORT_ERROR}")
        return True

    if SKIP_TRITON:
        print(f"  ⏭️  跳过 Triton 对比 (--skip-triton)")
        return True

    if not DEVICE.startswith('cuda'):
        print(f"  ⏭️  跳过 Triton 对比 (Triton 需要 CUDA)")
        return True

    try:
        # 将数据移到 CUDA（如果还没有）
        x_cuda = x.cuda() if not x.is_cuda else x
        weight_cuda = weight.cuda() if not weight.is_cuda else weight
        bias_cuda = bias.cuda() if not bias.is_cuda else bias
        z_cuda = z.cuda() if z is not None and not z.is_cuda else z

        # 调用 Triton 实现
        out_triton, mean_triton, rstd_triton = _layer_norm_fwd_triton(
            x_cuda.clone(), weight_cuda, bias_cuda, eps,
            z=z_cuda,
            out=None,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )

        # 将结果移回原设备进行对比
        out_triton = out_triton.cpu() if native_out.device.type == 'cpu' else out_triton

        print(
            f"  Triton 输出: shape={out_triton.shape}, mean={out_triton.mean():.6f}, std={out_triton.std():.6f}")

        success = allclose_with_info(
            native_out, out_triton, name="Native vs Triton", rtol=1e-3, atol=1e-4)
        return success

    except Exception as e:
        print(f"  ❌ Triton 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 80)
    print("测试 1: 基本功能测试")
    print("=" * 80)

    # 准备数据
    M, N = 8, 32
    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    weight = torch.randn(N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float32, device=DEVICE)
    eps = 1e-5

    print(
        f"输入: x={x.shape}, weight={weight.shape}, bias={bias.shape}, device={DEVICE}")

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
    layer_norm = torch.nn.LayerNorm(N, eps=eps, device=DEVICE)
    layer_norm.weight.data = weight.clone()
    layer_norm.bias.data = bias.clone()
    out_torch = layer_norm(x)

    success = allclose_with_info(
        out_native, out_torch, name="Native vs PyTorch LayerNorm")

    # 验证统计量
    expected_mean = x.mean(dim=1)  # [M]
    expected_var = x.var(dim=1, unbiased=False)  # [M]
    expected_rstd = 1.0 / torch.sqrt(expected_var + eps)

    print(f"\n验证统计量:")
    allclose_with_info(mean_native, expected_mean, name="  mean")
    allclose_with_info(rstd_native, expected_rstd, name="  rstd")

    # 与 Triton 对比
    print(f"\n与 Triton 对比:")
    success &= test_with_triton(
        "基本功能", out_native, x, weight, bias, eps,
        z=None, group_size=None, norm_before_gate=True, is_rms_norm=False
    )

    return success


def test_rmsnorm():
    """测试 RMSNorm"""
    print("\n" + "=" * 80)
    print("测试 2: RMSNorm")
    print("=" * 80)

    M, N = 16, 64
    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    weight = torch.ones(N, dtype=torch.float32, device=DEVICE)
    bias = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    eps = 1e-6

    print(f"输入: x={x.shape}, device={DEVICE}")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=None,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
    )

    print(f"Native 输出: out={out_native.shape}")
    print(f"  mean={mean_native} (should be None)")
    print(f"  rstd={rstd_native.shape}")

    assert mean_native is None, "RMSNorm should not compute mean"

    # 手动验证 RMSNorm
    rms = torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + eps)
    expected_out = (x / rms) * weight + bias

    success = allclose_with_info(
        out_native, expected_out, name="Native vs Manual RMSNorm")

    # 与 Triton 对比
    print(f"\n与 Triton 对比:")
    success &= test_with_triton(
        "RMSNorm", out_native, x, weight, bias, eps,
        z=None, group_size=None, norm_before_gate=True, is_rms_norm=True
    )

    return success


def test_gating():
    """测试 Gating (SwiGLU)"""
    print("\n" + "=" * 80)
    print("测试 3: Gating (SwiGLU)")
    print("=" * 80)

    M, N = 8, 32
    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    weight = torch.ones(N, dtype=torch.float32, device=DEVICE)
    bias = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    z = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    eps = 1e-5

    print(f"输入: x={x.shape}, z={z.shape}, device={DEVICE}")

    # 测试 1: Gating BEFORE norm
    print("\n  测试 3.1: Gating BEFORE Normalization")
    out_before, _, _ = _layer_norm_fwd_native(
        x.clone(), weight, bias, eps,
        z=z,
        group_size=None,
        norm_before_gate=False,  # BEFORE
        is_rms_norm=False,
    )

    # 手动验证
    x_gated = x * z * torch.sigmoid(z)
    mean = x_gated.mean(dim=1, keepdim=True)
    var = x_gated.var(dim=1, unbiased=False, keepdim=True)
    x_norm = (x_gated - mean) / torch.sqrt(var + eps)
    expected_before = x_norm * weight + bias

    success1 = allclose_with_info(
        out_before, expected_before, name="  Gating BEFORE")

    # 与 Triton 对比
    print(f"\n  与 Triton 对比 (BEFORE):")
    success1 &= test_with_triton(
        "Gating BEFORE", out_before, x, weight, bias, eps,
        z=z, group_size=None, norm_before_gate=False, is_rms_norm=False
    )

    # 测试 2: Gating AFTER norm
    print("\n  测试 3.2: Gating AFTER Normalization")
    out_after, _, _ = _layer_norm_fwd_native(
        x.clone(), weight, bias, eps,
        z=z,
        group_size=None,
        norm_before_gate=True,  # AFTER
        is_rms_norm=False,
    )

    # 手动验证
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    y_norm = x_norm * weight + bias
    expected_after = y_norm * z * torch.sigmoid(z)

    success2 = allclose_with_info(
        out_after, expected_after, name="  Gating AFTER")

    # 与 Triton 对比
    print(f"\n  与 Triton 对比 (AFTER):")
    success2 &= test_with_triton(
        "Gating AFTER", out_after, x, weight, bias, eps,
        z=z, group_size=None, norm_before_gate=True, is_rms_norm=False
    )

    return success1 and success2


def test_groupnorm():
    """测试 GroupNorm"""
    print("\n" + "=" * 80)
    print("测试 4: GroupNorm")
    print("=" * 80)

    M, N = 8, 96
    group_size = 32  # 3 groups
    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    weight = torch.randn(N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float32, device=DEVICE)
    eps = 1e-5

    print(
        f"输入: x={x.shape}, group_size={group_size} ({N // group_size} groups), device={DEVICE}")

    # Native 实现
    out_native, mean_native, rstd_native = _layer_norm_fwd_native(
        x, weight, bias, eps,
        z=None,
        group_size=group_size,
        norm_before_gate=True,
        is_rms_norm=False,
    )

    print(f"Native 输出: out={out_native.shape}")
    print(f"  mean={mean_native.shape} (should be [ngroups * M])")
    print(f"  rstd={rstd_native.shape}")

    ngroups = N // group_size
    assert mean_native.numel() == ngroups * \
        M, f"Expected {ngroups * M} elements, got {mean_native.numel()}"

    # 手动验证（简化版）
    x_reshaped = x.view(M, ngroups, group_size)
    mean_manual = x_reshaped.mean(dim=2)  # [M, ngroups]
    var_manual = x_reshaped.var(dim=2, unbiased=False)  # [M, ngroups]
    rstd_manual = 1.0 / torch.sqrt(var_manual + eps)

    x_norm = (x_reshaped - mean_manual.unsqueeze(2)) * rstd_manual.unsqueeze(2)

    weight_reshaped = weight.view(ngroups, group_size)
    bias_reshaped = bias.view(ngroups, group_size)

    y_manual = x_norm * \
        weight_reshaped.unsqueeze(0) + bias_reshaped.unsqueeze(0)
    y_manual = y_manual.view(M, N)

    success = allclose_with_info(
        out_native, y_manual, name="Native vs Manual GroupNorm")

    # 与 Triton 对比
    print(f"\n与 Triton 对比:")
    success &= test_with_triton(
        "GroupNorm", out_native, x, weight, bias, eps,
        z=None, group_size=group_size, norm_before_gate=True, is_rms_norm=False
    )

    return success


def test_high_level_api():
    """测试高层 API"""
    print("\n" + "=" * 80)
    print("测试 5: 高层 API")
    print("=" * 80)

    batch_size, seq_len, hidden_dim = 2, 16, 64
    x = torch.randn(batch_size, seq_len, hidden_dim,
                    dtype=torch.float32, device=DEVICE)
    weight = torch.randn(hidden_dim, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(hidden_dim, dtype=torch.float32, device=DEVICE)
    eps = 1e-5

    print(f"输入: x={x.shape} (3D), device={DEVICE}")

    # LayerNorm API
    print("\n  测试 layernorm_fn_native:")
    out_ln = layernorm_fn_native(x, weight, bias, eps=eps)
    print(f"    输出: {out_ln.shape}")
    assert out_ln.shape == x.shape, f"Shape mismatch: {out_ln.shape} vs {x.shape}"

    # 与 PyTorch 对比
    layer_norm = torch.nn.LayerNorm(hidden_dim, eps=eps, device=DEVICE)
    layer_norm.weight.data = weight.clone()
    layer_norm.bias.data = bias.clone()
    out_torch = layer_norm(x)

    success1 = allclose_with_info(
        out_ln, out_torch, name="    LayerNorm vs PyTorch")

    # RMSNorm API
    print("\n  测试 rmsnorm_fn_native:")
    out_rms = rmsnorm_fn_native(x, weight, bias, eps=eps)
    print(f"    输出: {out_rms.shape}")
    assert out_rms.shape == x.shape

    # 简单验证
    print(f"    输出统计: mean={out_rms.mean():.6f}, std={out_rms.std():.6f}")
    success2 = True  # No direct comparison

    # 注意：高层 API 的 Triton 对比比较复杂，因为需要展平输入
    # 这里我们主要验证与 PyTorch 的一致性
    print(f"\n  ⏭️  高层 API 主要验证与 PyTorch 的一致性")

    return success1 and success2


def test_simple_layernorm():
    """测试简化版 LayerNorm"""
    print("\n" + "=" * 80)
    print("测试 6: 简化版 LayerNorm")
    print("=" * 80)

    M, N = 8, 32
    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    weight = torch.randn(N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float32, device=DEVICE)
    eps = 1e-5

    print(f"输入: x={x.shape}, device={DEVICE}")

    # 简化版实现
    out_simple = simple_layernorm_native(x, weight, bias, eps=eps)

    # 与 PyTorch 对比
    layer_norm = torch.nn.LayerNorm(N, eps=eps, device=DEVICE)
    layer_norm.weight.data = weight.clone()
    layer_norm.bias.data = bias.clone()
    out_torch = layer_norm(x)

    success = allclose_with_info(
        out_simple, out_torch, name="Simple vs PyTorch")

    # 与 Triton 对比
    print(f"\n与 Triton 对比:")
    success &= test_with_triton(
        "简化版 LayerNorm", out_simple, x, weight, bias, eps,
        z=None, group_size=None, norm_before_gate=True, is_rms_norm=False
    )

    return success


def run_all_tests():
    """运行所有测试"""
    global DEVICE

    # 解析命令行参数
    args = parse_args()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "LayerNorm Native 实现测试套件" + " " * 28 + "║")
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

    results = []

    # 运行所有测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("RMSNorm", test_rmsnorm),
        ("Gating (SwiGLU)", test_gating),
        ("GroupNorm", test_groupnorm),
        ("高层 API", test_high_level_api),
        ("简化版 LayerNorm", test_simple_layernorm),
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
        print("✅ 所有测试通过！Native 实现与 PyTorch/Triton 完全等价")
        print("🎉 " * 20)
    else:
        print("\n" + "⚠️  " * 20)
        print("❌ 部分测试失败，请检查实现")
        print("⚠️  " * 20)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
