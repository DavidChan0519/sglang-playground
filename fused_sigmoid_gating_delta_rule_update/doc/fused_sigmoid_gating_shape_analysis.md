# `fused_sigmoid_gating_delta_rule_update` 输出形状分析

## 概述

本文档详细分析 `fused_sigmoid_gating_delta_rule_update` 函数的实现，追踪输入到输出的形状变化过程。

---

## 函数签名和关键代码

### 函数定义
```python
def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,          # query
    k: torch.Tensor,          # key
    v: torch.Tensor,          # value
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
```

### 关键代码片段（第 187-232 行）

```python
# 第 187 行：从输入提取形状信息
B, T, H, K, V = *k.shape, v.shape[-1]
HV = v.shape[2]
N = B if cu_seqlens is None else len(cu_seqlens) - 1

# 第 201 行：创建输出张量
o = q.new_empty(NK, *v.shape)

# 第 231-232 行：返回结果
o = o.squeeze(0)
return o
```

---

## 输入形状分析

### 从调用点追踪输入形状（hybrid_linear_attn_backend.py 第 260-265 行）

```python
# 第 260-265 行：输入重塑
seq_len = query.shape[0]
num_heads = query.shape[1] // head_k_dim
query = query.view(1, seq_len, num_heads, head_k_dim)
key = key.view(1, seq_len, num_heads, head_k_dim)
value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)
```

**输入形状**：
- `query`: `[1, seq_len, num_heads, head_k_dim]` = `[B, T, H, K]`
- `key`: `[1, seq_len, num_heads, head_k_dim]` = `[B, T, H, K]`
- `value`: `[1, seq_len, HV, head_v_dim]` = `[B, T, HV, V]`

### 形状变量定义（第 187-192 行）

```python
B, T, H, K, V = *k.shape, v.shape[-1]
# B: batch_size = 1 (decode 模式下通常是 1)
# T: seq_len (序列长度)
# H: num_heads (attention heads 数量)
# K: head_k_dim (每个 head 的 key 维度)
# V: v.shape[-1] = head_v_dim (value 的最后一个维度)

HV = v.shape[2]
# HV: value 张量的第 3 个维度，表示 value heads 的数量

N = B if cu_seqlens is None else len(cu_seqlens) - 1
# N: 有效的批次数量

BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
# BK: K 的下一个 2 的幂次 (Triton block size for K)
# BV: min(V 的下一个 2 的幂次, 8) (Triton block size for V)

NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
# NK: K 维度需要的 block 数量 (通常是 1，代码中有 assert)
# NV: V 维度需要的 block 数量

assert NK == 1, "NK > 1 is not supported yet"
```

---

## 输出形状计算过程

### 第 1 步：创建输出张量（第 201 行）

```python
o = q.new_empty(NK, *v.shape)
```

**分析**：
- `NK = 1` (已断言)
- `*v.shape` 解包为 `B, T, HV, V`
- 因此：`o.shape = [1, B, T, HV, V]`

**具体值**：
- `o.shape = [1, 1, seq_len, HV, head_v_dim]`

### 第 2 步：Squeeze 操作（第 231 行）

```python
o = o.squeeze(0)
```

**分析**：
- `squeeze(0)` 移除第 0 维（大小为 1 的维度）
- 结果：`o.shape = [B, T, HV, V]`

**具体值**：
- `o.shape = [1, seq_len, HV, head_v_dim]`

---

## 最终输出形状

### 返回值 `core_attn_out` 的形状

```
core_attn_out.shape = [B, T, HV, V]
                    = [1, seq_len, HV, head_v_dim]
                    = [batch_size, sequence_length, value_heads, head_value_dim]
```

### 维度含义

| 维度 | 符号 | 含义 | 示例值 |
|------|------|------|--------|
| 0 | `B` | Batch size | `1` (decode 模式) |
| 1 | `T` | Sequence length | 可变（如 64, 128, 256） |
| 2 | `HV` | Value heads 数量 | 取决于模型配置 |
| 3 | `V` | 每个 value head 的维度 | `head_v_dim` |

---

## Triton Kernel 实现细节

### Grid 配置（第 202 行）

```python
grid = (NK, NV, N * HV)
```

**说明**：
- 第 1 维：`NK = 1` (K 维度的 block 数)
- 第 2 维：`NV = triton.cdiv(V, BV)` (V 维度的 block 数)
- 第 3 维：`N * HV` (batch × value_heads)

### Kernel 输出存储（kernel 第 69, 141 行）

```python
# 第 69 行：输出指针初始化
p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

# 第 141 行：存储计算结果
b_o = tl.sum(b_h * b_q[:, None], 0)  # 计算输出 [BV]
tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
```

**计算逻辑**：
- 对每个时间步 `t`，计算 `o[t] = sum(h * q, dim=K)` 
- `h` 是隐藏状态 `[K, V]`
- `q` 是查询 `[K]`
- 结果 `o[t]` 的形状是 `[V]`
- 所有时间步的输出拼接后：`[T, HV, V]` (单个 batch 的情况)
- 加上 batch 维度：`[B, T, HV, V]`

---

## 实际数值示例

假设：
- `batch_size = 1`
- `seq_len = 64`
- `num_heads = 8`
- `head_k_dim = 128`
- `head_v_dim = 64`
- `HV = 8` (value heads)

### 输入形状：
```python
query.shape  = [1, 64, 8, 128]   # [B, T, H, K]
key.shape    = [1, 64, 8, 128]   # [B, T, H, K]
value.shape  = [1, 64, 8, 64]    # [B, T, HV, V]
```

### 中间计算：
```python
B=1, T=64, H=8, K=128, V=64, HV=8
NK=1 (因为 K=128, BK=128, NK=ceil(128/128)=1)
NV=ceil(64/8)=8 (因为 V=64, BV=8)
```

### 输出形状：
```python
# 第 201 行创建
o.shape = [1, 1, 64, 8, 64]  # [NK, B, T, HV, V]

# 第 231 行 squeeze
o.shape = [1, 64, 8, 64]     # [B, T, HV, V]

# 最终返回
core_attn_out.shape = [1, 64, 8, 64]
```

---

## 形状变化流程图

```
输入阶段:
  query:  [B=1, T, H, K]
  key:    [B=1, T, H, K]
  value:  [B=1, T, HV, V]
       ↓
提取形状参数:
  B=1, T, H, K, V, HV
  NK=1, NV
       ↓
创建输出张量:
  o = [NK=1, B=1, T, HV, V]
       ↓
Triton Kernel 计算:
  Grid: (NK=1, NV, N*HV)
  每个程序计算一个 (block_K, block_V, batch*value_head) 的输出
       ↓
Squeeze 第 0 维:
  o = [B=1, T, HV, V]
       ↓
返回结果:
  core_attn_out = [B=1, T, HV, V]
```

---

## 总结

### 🎯 核心结论

**`core_attn_out` 的形状为 `[B, T, HV, V]`**

在 decode 模式下的典型值：
```python
core_attn_out.shape = [1, seq_len, value_heads, head_v_dim]
```

### 关键点

1. **输入形状**：函数接收 `[B, T, H, K]` 的 query/key 和 `[B, T, HV, V]` 的 value

2. **中间创建**：首先创建 `[NK, B, T, HV, V]` 的输出张量，其中 `NK=1`

3. **Squeeze 操作**：移除第 0 维，得到最终形状 `[B, T, HV, V]`

4. **物理意义**：
   - `B`: 批次大小
   - `T`: 序列长度（每个 token）
   - `HV`: Value heads 数量
   - `V`: 每个 value head 的特征维度

5. **与输入对应**：
   - 输出的 `T` 维度与输入 query/key/value 的 `T` 维度一致
   - 输出的 `HV, V` 维度与输入 value 的 `HV, V` 维度一致
   - 输出的 `B` 维度与输入的 `B` 维度一致

---

最后更新: 2025-01-17

