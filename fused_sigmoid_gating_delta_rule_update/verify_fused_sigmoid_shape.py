#!/usr/bin/env python3
"""
验证 fused_sigmoid_gating_delta_rule_update 的输出形状

这个脚本通过模拟函数的形状计算逻辑，验证输出形状的推导过程。
"""


def analyze_output_shape(B, T, H, K, V, HV):
    """
    分析 fused_sigmoid_gating_delta_rule_update 的输出形状

    参数:
        B: batch_size
        T: sequence_length
        H: num_attention_heads
        K: head_k_dim (key dimension per head)
        V: head_v_dim (value dimension per head)
        HV: num_value_heads

    返回:
        输出张量的形状
    """
    print(f"\n{'='*80}")
    print(f"输入参数分析")
    print(f"{'='*80}")
    print(f"B (batch_size)           : {B}")
    print(f"T (sequence_length)      : {T}")
    print(f"H (num_attention_heads)  : {H}")
    print(f"K (head_k_dim)           : {K}")
    print(f"V (head_v_dim)           : {V}")
    print(f"HV (num_value_heads)     : {HV}")

    # 模拟函数内部的形状计算
    print(f"\n{'='*80}")
    print(f"中间计算过程")
    print(f"{'='*80}")

    # 第 187 行：提取形状
    print(f"\n1. 从 k.shape 和 v.shape 提取参数:")
    k_shape = (B, T, H, K)
    v_shape = (B, T, HV, V)
    print(f"   k.shape = {k_shape}")
    print(f"   v.shape = {v_shape}")
    print(f"   提取: B={B}, T={T}, H={H}, K={K}, V={V}, HV={HV}")

    # 第 190-192 行：计算 Triton block 参数
    import math
    BK = 2 ** math.ceil(math.log2(K))  # next_power_of_2(K)
    BV = min(2 ** math.ceil(math.log2(V)), 8)  # min(next_power_of_2(V), 8)
    NK = math.ceil(K / BK)  # triton.cdiv(K, BK)
    NV = math.ceil(V / BV)  # triton.cdiv(V, BV)

    print(f"\n2. 计算 Triton block 参数:")
    print(f"   BK (block_K)  = next_power_of_2({K}) = {BK}")
    print(f"   BV (block_V)  = min(next_power_of_2({V}), 8) = {BV}")
    print(f"   NK (num blocks for K) = ceil({K}/{BK}) = {NK}")
    print(f"   NV (num blocks for V) = ceil({V}/{BV}) = {NV}")

    assert NK == 1, f"NK must be 1, but got {NK}"
    print(f"   ✓ 断言通过: NK == 1")

    # 第 201 行：创建输出张量
    print(f"\n3. 创建输出张量 (第 201 行):")
    print(f"   o = q.new_empty(NK, *v.shape)")
    print(f"   o = q.new_empty({NK}, {B}, {T}, {HV}, {V})")
    o_shape_initial = (NK, B, T, HV, V)
    print(f"   o.shape = {o_shape_initial}")

    # 第 202 行：Grid 配置
    N = B  # 假设没有 cu_seqlens
    grid = (NK, NV, N * HV)
    print(f"\n4. Triton Grid 配置 (第 202 行):")
    print(f"   grid = (NK={NK}, NV={NV}, N*HV={N}*{HV}={N*HV})")
    print(f"   grid = {grid}")
    print(f"   说明: 每个程序处理一个 (block_K, block_V, batch*value_head) 的输出块")

    # 第 231 行：Squeeze 操作
    print(f"\n5. Squeeze 操作 (第 231 行):")
    print(f"   o = o.squeeze(0)")
    print(f"   移除第 0 维 (大小为 {NK})")
    o_shape_final = (B, T, HV, V)
    print(f"   o.shape = {o_shape_final}")

    # 第 232 行：返回
    print(f"\n6. 返回结果 (第 232 行):")
    print(f"   return o")

    print(f"\n{'='*80}")
    print(f"最终输出形状")
    print(f"{'='*80}")
    print(f"core_attn_out.shape = {o_shape_final}")
    print(f"\n形状含义:")
    print(f"  维度 0: B={B}   - Batch size")
    print(f"  维度 1: T={T}   - Sequence length")
    print(f"  维度 2: HV={HV}  - Number of value heads")
    print(f"  维度 3: V={V}   - Head value dimension")

    return o_shape_final


def test_examples():
    """测试几个具体的例子"""

    print("\n" + "="*80)
    print("测试用例")
    print("="*80)

    # 测试用例 1: 小规模
    print("\n【测试 1】小规模配置")
    shape1 = analyze_output_shape(B=1, T=64, H=8, K=128, V=64, HV=8)

    # 测试用例 2: 中等规模
    print("\n\n【测试 2】中等规模配置")
    shape2 = analyze_output_shape(B=1, T=128, H=16, K=256, V=128, HV=16)

    # 测试用例 3: 大规模
    print("\n\n【测试 3】大规模配置")
    shape3 = analyze_output_shape(B=1, T=256, H=32, K=512, V=256, HV=32)

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("\n✓ 所有测试用例的输出形状公式:")
    print("  core_attn_out.shape = [B, T, HV, V]")
    print("\n✓ 输出维度始终等于输入 value 的维度:")
    print("  value.shape         = [B, T, HV, V]")
    print("  core_attn_out.shape = [B, T, HV, V]")
    print("\n✓ 关键观察:")
    print("  1. 输出保持了输入的 batch size (B)")
    print("  2. 输出保持了输入的 sequence length (T)")
    print("  3. 输出保持了 value 的 heads 数量 (HV)")
    print("  4. 输出保持了 value 的每个 head 的维度 (V)")
    print("  5. query 的 attention heads (H) 不出现在输出形状中")
    print("  6. key 的维度 (K) 不出现在输出形状中")


def compare_with_input():
    """比较输入和输出的形状关系"""

    print("\n" + "="*80)
    print("输入输出形状对比")
    print("="*80)

    B, T, H, K, V, HV = 1, 128, 16, 256, 128, 16

    print(f"\n假设参数: B={B}, T={T}, H={H}, K={K}, V={V}, HV={HV}")

    print(f"\n输入形状:")
    print(f"  query.shape = [{B}, {T}, {H}, {K}]")
    print(f"  key.shape   = [{B}, {T}, {H}, {K}]")
    print(f"  value.shape = [{B}, {T}, {HV}, {V}]")

    print(f"\n输出形状:")
    print(f"  core_attn_out.shape = [{B}, {T}, {HV}, {V}]")

    print(f"\n形状变换:")
    print(f"  [{B}, {T}, {H}, {K}] (query)")
    print(f"         +")
    print(f"  [{B}, {T}, {H}, {K}] (key)")
    print(f"         +")
    print(f"  [{B}, {T}, {HV}, {V}] (value)")
    print(f"         ↓")
    print(f"  fused_sigmoid_gating_delta_rule_update()")
    print(f"         ↓")
    print(f"  [{B}, {T}, {HV}, {V}] (output)")

    print(f"\n形状关系:")
    print(f"  ✓ 输出的前两个维度 [B, T] 与输入一致")
    print(f"  ✓ 输出的后两个维度 [HV, V] 与 value 的后两个维度一致")
    print(f"  ✓ query/key 的 H 和 K 维度在计算中被消耗掉")


if __name__ == "__main__":
    print("="*80)
    print("fused_sigmoid_gating_delta_rule_update 输出形状分析")
    print("="*80)

    # 运行测试用例
    test_examples()

    # 对比输入输出
    compare_with_input()

    print("\n" + "="*80)
    print("✓ 分析完成")
    print("="*80)
    print("\n📝 详细文档: fused_sigmoid_gating_shape_analysis.md")
