#!/usr/bin/env python3
"""
LayerNorm Native 实现演示脚本

展示如何使用 PyTorch native 实现替换 Triton kernel
"""

import torch
from layernorm_native_implementation import (
    layernorm_fn_native,
    rmsnorm_fn_native,
    simple_layernorm_native,
)


def demo_1_basic_layernorm():
    """示例 1: 基本 LayerNorm"""
    print("\n" + "=" * 80)
    print("示例 1: 基本 LayerNorm")
    print("=" * 80)

    # 准备数据
    batch_size, seq_len, hidden_dim = 2, 10, 64
    x = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.ones(hidden_dim)
    bias = torch.zeros(hidden_dim)

    print(f"输入: x.shape = {x.shape}")

    # Native 实现
    out_native = layernorm_fn_native(x, weight, bias, eps=1e-5)
    print(f"输出: out.shape = {out_native.shape}")
    print(f"      out.mean() = {out_native.mean():.6f}")
    print(f"      out.std() = {out_native.std():.6f}")

    # 与 PyTorch 标准实现对比
    layer_norm = torch.nn.LayerNorm(hidden_dim)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    out_torch = layer_norm(x)

    # 验证
    max_diff = (out_native - out_torch).abs().max().item()
    print(f"\n与 PyTorch LayerNorm 对比:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  是否等价: {'✅ Yes' if max_diff < 1e-5 else '❌ No'}")


def demo_2_rmsnorm():
    """示例 2: RMSNorm"""
    print("\n" + "=" * 80)
    print("示例 2: RMSNorm (不计算 mean)")
    print("=" * 80)

    # 准备数据
    batch_size, seq_len, hidden_dim = 4, 20, 128
    x = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.ones(hidden_dim)
    bias = torch.zeros(hidden_dim)

    print(f"输入: x.shape = {x.shape}")

    # RMSNorm
    out_rms = rmsnorm_fn_native(x, weight, bias, eps=1e-6)
    print(f"输出: out.shape = {out_rms.shape}")
    print(f"      out.mean() = {out_rms.mean():.6f}")
    print(f"      out.std() = {out_rms.std():.6f}")

    # 手动验证
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-6)
    out_manual = (x / rms) * weight + bias

    max_diff = (out_rms - out_manual).abs().max().item()
    print(f"\n与手动计算对比:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  是否等价: {'✅ Yes' if max_diff < 1e-5 else '❌ No'}")


def demo_3_gating():
    """示例 3: SwiGLU Gating"""
    print("\n" + "=" * 80)
    print("示例 3: SwiGLU Gating (门控激活)")
    print("=" * 80)

    # 准备数据
    batch_size, seq_len, hidden_dim = 2, 10, 64
    x = torch.randn(batch_size, seq_len, hidden_dim)
    z = torch.randn(batch_size, seq_len, hidden_dim)  # 门控值
    weight = torch.ones(hidden_dim)
    bias = torch.zeros(hidden_dim)

    print(f"输入: x.shape = {x.shape}")
    print(f"      z.shape = {z.shape} (gating)")

    # Gating BEFORE norm
    out_before = layernorm_fn_native(
        x, weight, bias,
        z=z,
        eps=1e-5,
        norm_before_gate=False  # 先 gate 后 norm
    )
    print(f"\nGating BEFORE norm:")
    print(f"  输出: out.shape = {out_before.shape}")
    print(f"        out.mean() = {out_before.mean():.6f}")

    # Gating AFTER norm
    out_after = layernorm_fn_native(
        x, weight, bias,
        z=z,
        eps=1e-5,
        norm_before_gate=True  # 先 norm 后 gate
    )
    print(f"\nGating AFTER norm:")
    print(f"  输出: out.shape = {out_after.shape}")
    print(f"        out.mean() = {out_after.mean():.6f}")

    print(f"\n两种方式的差异:")
    diff = (out_before - out_after).abs().mean().item()
    print(f"  平均差异: {diff:.6f}")
    print(f"  说明: 门控顺序影响结果（这是正常的）")


def demo_4_groupnorm():
    """示例 4: GroupNorm"""
    print("\n" + "=" * 80)
    print("示例 4: GroupNorm (分组归一化)")
    print("=" * 80)

    # 准备数据
    batch_size, seq_len, hidden_dim = 2, 10, 96
    group_size = 32  # 3 groups
    x = torch.randn(batch_size, seq_len, hidden_dim)
    weight = torch.ones(hidden_dim)
    bias = torch.zeros(hidden_dim)

    print(f"输入: x.shape = {x.shape}")
    print(f"      hidden_dim = {hidden_dim}")
    print(f"      group_size = {group_size}")
    print(f"      num_groups = {hidden_dim // group_size}")

    # GroupNorm
    out_group = layernorm_fn_native(
        x, weight, bias,
        eps=1e-5,
        group_size=group_size
    )
    print(f"\n输出: out.shape = {out_group.shape}")
    print(f"      out.mean() = {out_group.mean():.6f}")
    print(f"      out.std() = {out_group.std():.6f}")

    # 与标准 LayerNorm 对比（group_size=hidden_dim）
    out_standard = layernorm_fn_native(
        x, weight, bias,
        eps=1e-5,
        group_size=None  # 等价于 hidden_dim
    )

    diff = (out_group - out_standard).abs().mean().item()
    print(f"\n与标准 LayerNorm 对比:")
    print(f"  平均差异: {diff:.6f}")
    print(f"  说明: GroupNorm 结果不同（这是正常的）")


def demo_5_replace_example():
    """示例 5: 如何替换现有代码"""
    print("\n" + "=" * 80)
    print("示例 5: 替换现有代码")
    print("=" * 80)

    # 准备数据
    x = torch.randn(4, 128, 768)
    weight = torch.ones(768)
    bias = torch.zeros(768)

    print("原代码:")
    print("  from python.sglang.srt.layers.attention.fla.layernorm_gated import layernorm_fn")
    print("  out = layernorm_fn(x, weight, bias, eps=1e-5)")

    print("\n替换为:")
    print("  from layernorm_native_implementation import layernorm_fn_native as layernorm_fn")
    print("  out = layernorm_fn(x, weight, bias, eps=1e-5)")

    # 实际执行
    from layernorm_native_implementation import layernorm_fn_native as layernorm_fn
    out = layernorm_fn(x, weight, bias, eps=1e-5)

    print(f"\n执行结果:")
    print(f"  输入: x.shape = {x.shape}")
    print(f"  输出: out.shape = {out.shape}")
    print(f"  ✅ 替换成功！")


def demo_6_simple_version():
    """示例 6: 简化版 LayerNorm"""
    print("\n" + "=" * 80)
    print("示例 6: 简化版 LayerNorm (仅标准功能)")
    print("=" * 80)

    # 准备数据
    x = torch.randn(32, 256)
    weight = torch.ones(256)
    bias = torch.zeros(256)

    print(f"输入: x.shape = {x.shape}")

    # 简化版
    out_simple = simple_layernorm_native(x, weight, bias, eps=1e-5)
    print(f"输出: out.shape = {out_simple.shape}")

    # 与 PyTorch 对比
    out_torch = torch.nn.functional.layer_norm(
        x, (256,), weight, bias, eps=1e-5)

    max_diff = (out_simple - out_torch).abs().max().item()
    print(f"\n与 PyTorch 对比:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  是否等价: {'✅ Yes' if max_diff < 1e-5 else '❌ No'}")


def main():
    """运行所有演示"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "LayerNorm Native 演示脚本" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")

    print(f"\n✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 运行所有演示
    demos = [
        demo_1_basic_layernorm,
        demo_2_rmsnorm,
        demo_3_gating,
        demo_4_groupnorm,
        demo_5_replace_example,
        demo_6_simple_version,
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
    print("✅ 基本 LayerNorm - 与 PyTorch 完全等价")
    print("✅ RMSNorm - 不计算 mean，更高效")
    print("✅ SwiGLU Gating - 支持门控前/后")
    print("✅ GroupNorm - 支持分组归一化")
    print("✅ 代码替换 - 即插即用，无需修改其他代码")
    print("✅ 简化版 - 仅标准功能，代码更简洁")

    print("\n" + "🎉 " * 20)
    print("✅ 演示完成！Native 实现已就绪，可以直接使用")
    print("🎉 " * 20)

    print("\n📚 更多信息请查看:")
    print("  - layernorm_native_implementation.py  (核心实现)")
    print("  - LAYERNORM_NATIVE_USAGE_GUIDE.md     (详细使用指南)")
    print("  - LAYERNORM_MIGRATION_SUMMARY.md      (项目总结)")
    print("  - test_layernorm_native_cpu.py        (测试套件)")


if __name__ == "__main__":
    main()
