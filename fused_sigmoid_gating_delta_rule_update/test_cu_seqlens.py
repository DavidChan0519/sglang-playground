#!/usr/bin/env python3
"""
测试 cu_seqlens (变长序列) 功能

验证 Native 实现对变长序列批处理的支持

使用方法:
    # CPU 测试
    python3 test_cu_seqlens.py --device cpu
    
    # CUDA 测试
    python3 test_cu_seqlens.py --device cuda
    
    # 跳过 Triton 对比
    python3 test_cu_seqlens.py --device cuda --skip-triton
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='测试 cu_seqlens (变长序列) 功能')
    parser.add_argument('--device', type=str, default='cpu',
                        help='测试设备 (cpu, cuda, cuda:0, etc.). 默认: cpu')
    parser.add_argument('--skip-triton', action='store_true',
                        help='跳过 Triton 对比测试')
    args = parser.parse_args()

    global DEVICE, SKIP_TRITON
    DEVICE = args.device
    SKIP_TRITON = args.skip_triton

    return args


def test_varlen_vs_fixed_length():
    """测试：验证变长序列和固定长度序列的等价性"""
    print("\n" + "=" * 80)
    print("测试 1: 变长序列 vs 固定长度")
    print("=" * 80)

    dtype = torch.float32

    # 创建3个不同长度的序列
    # 序列 0: 长度 5
    # 序列 1: 长度 7
    # 序列 2: 长度 6
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
        # [1, seq_len, HV] -> [seq_len, HV]
        a_fixed = a_varlen[:, bos:eos].contiguous()
        q_fixed = q_varlen[:, bos:eos].contiguous()  # [1, seq_len, H, K]
        k_fixed = k_varlen[:, bos:eos].contiguous()  # [1, seq_len, H, K]
        v_fixed = v_varlen[:, bos:eos].contiguous()  # [1, seq_len, HV, V]
        b_fixed = b_varlen[:, bos:eos].contiguous()  # [1, seq_len, HV]

        # Native 固定长度模式
        out_native_fixed = fused_sigmoid_gating_delta_rule_update_native(
            A_log_varlen.clone(), a_fixed.clone(), dt_bias_varlen.clone(),
            softplus_beta, softplus_threshold,
            q_fixed.clone(), k_fixed.clone(), v_fixed.clone(), b_fixed.clone(),
            initial_state_source=None, initial_state_indices=None,
            scale=None, use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,  # 固定长度模式
        )

        # 从变长输出中提取对应部分
        # [seq_len, HV, V]
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

    # Triton 对比（如果可用）
    if HAS_TRITON and not SKIP_TRITON:
        print("\n[4] Triton 实现（变长模式）")
        try:
            out_triton_varlen = fused_sigmoid_gating_delta_rule_update_triton(
                A_log_varlen.clone(), a_varlen.clone(), dt_bias_varlen.clone(),
                softplus_beta, softplus_threshold,
                q_varlen.clone(), k_varlen.clone(), v_varlen.clone(), b_varlen.clone(),
                initial_state_source=None, initial_state_indices=None,
                scale=None, use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
            print(
                f"输出: shape={out_triton_varlen.shape}, mean={out_triton_varlen.mean():.6f}")

            max_diff_triton = (out_native_varlen -
                               out_triton_varlen).abs().max().item()
            print(f"\n✅ Native vs Triton (变长模式): 最大差异 {max_diff_triton:.2e}")

            if max_diff_triton > 1e-3:
                print(f"⚠️  注意: 差异较大，可能需要进一步检查")
                success = False
        except Exception as e:
            print(f"❌ Triton 测试失败: {e}")
            import traceback
            traceback.print_exc()
            success = False
    elif not HAS_TRITON:
        print(f"\n⚠️  跳过 Triton 对比: {TRITON_IMPORT_ERROR}")
    elif SKIP_TRITON:
        print(f"\n⏭️  跳过 Triton 对比 (--skip-triton)")

    return success


def test_varlen_with_initial_states():
    """测试：变长序列 + 初始状态"""
    print("\n" + "=" * 80)
    print("测试 2: 变长序列 + 初始状态")
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

    return success


def run_all_tests():
    """运行所有测试"""
    global DEVICE

    # 解析命令行参数
    args = parse_args()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "cu_seqlens 变长序列测试" + " " * 28 + "║")
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

    # 测试 1: 变长 vs 固定长度
    try:
        success = test_varlen_vs_fixed_length()
        results.append(("变长 vs 固定长度", success))
    except Exception as e:
        print(f"\n  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("变长 vs 固定长度", False))

    # 测试 2: 变长 + 初始状态
    try:
        success = test_varlen_with_initial_states()
        results.append(("变长 + 初始状态", success))
    except Exception as e:
        print(f"\n  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("变长 + 初始状态", False))

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
        print("✅ 所有 cu_seqlens 测试通过！")
        print("🎉 " * 20)
    else:
        print("\n" + "⚠️  " * 20)
        print("❌ 部分测试失败")
        print("⚠️  " * 20)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
