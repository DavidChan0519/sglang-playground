# `_layer_norm_fwd_1pass_kernel` 输出形状分析

## 概述

本文档详细分析 `_layer_norm_fwd_1pass_kernel` Triton kernel 的实现，追踪输入到输出的形状变化过程。

---

## 函数调用链

### 完整调用链

```
用户调用 LayerNorm.forward(x, z)  或  layernorm_fn(x, ...)
    ↓
LayerNormFn.apply(x, weight, bias, z, ...)
    ↓
LayerNormFn.forward(ctx, x, weight, bias, z, ...)
    ↓
_layer_norm_fwd(x, weight, bias, eps, z, ...)
    ↓
_layer_norm_fwd_1pass_kernel[grid](X, Y, W, B, Z, ...)
```

---

## 关键代码分析

### 第 1 步：`LayerNormFn.forward` - 输入预处理（第 200-209 行）

```python
x_shape_og = x.shape              # 保存原始形状
x = x.reshape(-1, x.shape[-1])    # 重塑为 2D: [*, N] -> [M, N]

if z is not None:
    z = z.reshape(-1, z.shape[-1])  # z 也重塑为 [M, N]
```

**分析**：
- `x_shape_og`：原始输入形状，可能是任意维度 `[d1, d2, ..., dk, N]`
- `x`：重塑后为 2D 张量 `[M, N]`，其中 `M = d1 * d2 * ... * dk`
- `N`：最后一个维度（特征维度）

**示例**：
```python
# 如果输入 x.shape = [batch_size, seq_len, hidden_dim]
# 那么重塑后:
M = batch_size * seq_len
N = hidden_dim
x.shape = [M, N]
```

### 第 2 步：`_layer_norm_fwd` - 创建输出张量（第 127-145 行）

```python
M, N = x.shape                    # 第 127 行：提取形状
if group_size is None:
    group_size = N                # 第 129 行：默认 group_size = N
ngroups = N // group_size         # 第 131 行：计算组数

# 第 142-145 行：创建输出张量
if out is not None:
    assert out.shape == x.shape   # 断言：输出形状必须等于输入形状
else:
    out = torch.empty_like(x)     # 创建与 x 形状相同的输出
```

**关键点**：
- ✅ **输出 `out` 的形状 = 输入 `x` 的形状 = `[M, N]`**
- `ngroups`：特征维度被分成多少组（用于 GroupNorm）
- 如果 `group_size = N`（默认），则 `ngroups = 1`（标准 LayerNorm）

### 第 3 步：Triton Kernel 配置（第 160 行）

```python
grid = (M, ngroups)
```

**说明**：
- `grid` 的第 1 维：`M` - 有多少行需要处理
- `grid` 的第 2 维：`ngroups` - 每行有多少组
- 每个 Triton 程序处理输入的一个 `(row, group)` 块

### 第 4 步：Triton Kernel 实现（第 53-113 行）

#### Kernel 签名
```python
def _layer_norm_fwd_1pass_kernel(
    X,              # 输入指针
    Y,              # 输出指针
    W,              # 权重指针
    B,              # bias 指针
    Z,              # gating 指针 (可选)
    Mean,           # mean 指针 (仅 LayerNorm，RMSNorm 不需要)
    Rstd,           # 1/std 指针
    stride_x_row,   # 输入行步长
    stride_y_row,   # 输出行步长
    stride_z_row,   # z 行步长
    M,              # 行数
    N,              # 列数 (group_size)
    eps,            # epsilon
    BLOCK_N: tl.constexpr,
    ...
):
```

#### Kernel 执行逻辑

```python
# 第 74-75 行：获取当前处理的行和组
row = tl.program_id(0)     # 0 到 M-1
group = tl.program_id(1)   # 0 到 ngroups-1

# 第 76-77 行：计算输入和输出的偏移
X += row * stride_x_row + group * N
Y += row * stride_y_row + group * N

# 第 87-88 行：加载输入数据 (一个 group 的数据)
cols = tl.arange(0, BLOCK_N)
x = tl.load(X + cols, mask=cols < N, other=0.0)

# 第 92-100 行：计算 mean 和 rstd
# ... (LayerNorm 或 RMSNorm 的计算逻辑)

# 第 102-108 行：归一化和线性变换
w = tl.load(W + cols, mask=mask)
x_hat = (x - mean) * rstd  # 或 x * rstd (RMSNorm)
y = x_hat * w + b          # 或 x_hat * w (无 bias)

# 第 109-111 行：可选的 gating
if HAS_Z and NORM_BEFORE_GATE:
    z = tl.load(Z + cols, mask=mask)
    y *= z * tl.sigmoid(z)  # Swish/SiLU gating

# 第 113 行：存储输出
tl.store(Y + cols, y, mask=mask)
```

**关键观察**：
- 每个 `(row, group)` 程序处理输入的一个片段（长度为 `group_size`）
- 输出写入与输入对应的位置
- **输出的形状完全由输入决定，没有维度变化**

### 第 5 步：返回结果（第 181 行）

```python
return out, mean, rstd
```

**返回值**：
- `out`：形状 `[M, N]` - 归一化后的输出
- `mean`：形状 `[ngroups * M]` - 每个组的均值（仅 LayerNorm）
- `rstd`：形状 `[ngroups * M]` - 每个组的 1/std

### 第 6 步：恢复原始形状（第 223 行）

```python
return y.reshape(x_shape_og)
```

**最终输出**：
- 从 2D 形状 `[M, N]` 恢复到原始形状 `x_shape_og`

---

## 输出形状总结

### 🎯 核心结论

**在 `_layer_norm_fwd` 函数中，输出 `out` 的形状为 `[M, N]`**

**在 `LayerNormFn.forward` 中，最终输出的形状为 `x_shape_og`（原始输入形状）**

### 形状对应关系

| 层级 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| `LayerNormFn.forward` (输入) | `[d1, d2, ..., dk, N]` | - | 原始输入，任意维度 |
| 重塑后 | `[M, N]` | - | `M = d1 * d2 * ... * dk` |
| `_layer_norm_fwd` (输出) | - | `[M, N]` | 与输入相同 |
| `LayerNormFn.forward` (输出) | - | `[d1, d2, ..., dk, N]` | 恢复原始形状 |

### 关键点

1. **形状保持不变**：`out.shape == x.shape`
2. **在 2D 处理**：内部将输入重塑为 `[M, N]` 进行处理
3. **恢复原始维度**：最终输出恢复到原始输入的形状
4. **特征维度不变**：最后一个维度 `N` 始终保持不变

---

## 实际示例

### 示例 1：标准 LayerNorm（3D 输入）

```python
# 输入
x = torch.randn(8, 128, 768)  # [batch_size, seq_len, hidden_dim]
weight = torch.ones(768)
bias = torch.zeros(768)

# 内部处理
M, N = 8 * 128, 768           # M=1024, N=768
x_2d = x.reshape(1024, 768)   # [M, N]

# Kernel 执行
grid = (1024, 1)               # (M, ngroups=1)
out_2d = torch.empty(1024, 768)  # [M, N]

# 最终输出
out = out_2d.reshape(8, 128, 768)  # [batch_size, seq_len, hidden_dim]
```

**输出形状**：`[8, 128, 768]` ✅ 与输入相同

### 示例 2：GroupNorm（4D 输入）

```python
# 输入
x = torch.randn(4, 64, 32, 512)  # [batch, height, width, channels]
weight = torch.ones(512)
bias = torch.zeros(512)
group_size = 128  # 每组 128 个特征

# 内部处理
M, N = 4 * 64 * 32, 512        # M=8192, N=512
ngroups = 512 // 128            # ngroups=4
x_2d = x.reshape(8192, 512)     # [M, N]

# Kernel 执行
grid = (8192, 4)                # (M, ngroups=4)
out_2d = torch.empty(8192, 512)  # [M, N]

# 最终输出
out = out_2d.reshape(4, 64, 32, 512)  # [batch, height, width, channels]
```

**输出形状**：`[4, 64, 32, 512]` ✅ 与输入相同

### 示例 3：RMSNorm with Gating（2D 输入）

```python
# 输入
x = torch.randn(1024, 2048)    # [tokens, hidden_dim]
z = torch.randn(1024, 2048)    # gating 分支
weight = torch.ones(2048)
bias = None  # RMSNorm 通常没有 bias

# 内部处理
M, N = 1024, 2048               # 已经是 2D
x_2d = x                        # 无需重塑

# Kernel 执行
grid = (1024, 1)                # (M, ngroups=1)
out_2d = torch.empty(1024, 2048)  # [M, N]

# 最终输出
out = out_2d                    # [tokens, hidden_dim]
```

**输出形状**：`[1024, 2048]` ✅ 与输入相同

---

## Grid 配置详解

### Grid 维度

```python
grid = (M, ngroups)
```

| Grid 维度 | 大小 | 含义 | 示例 |
|-----------|------|------|------|
| `grid[0]` | `M` | 输入的行数 | `batch_size * seq_len` |
| `grid[1]` | `ngroups` | 组数 | `N // group_size` |

### 不同场景下的 Grid

#### 场景 1：标准 LayerNorm
```python
group_size = N  # 默认
ngroups = 1
grid = (M, 1)
# 每行作为一个整体进行归一化
```

#### 场景 2：GroupNorm
```python
group_size = 128  # 自定义
N = 512
ngroups = 4
grid = (M, 4)
# 每行分成 4 组，每组独立归一化
```

### Kernel 程序分配

```python
# Triton 启动 M * ngroups 个程序
total_programs = M * ngroups

# 每个程序处理:
# - 第 row 行 (row = program_id(0))
# - 第 group 组 (group = program_id(1))
# - 共 group_size 个元素
```

---

## 内存布局分析

### 输入输出指针偏移

```python
# 第 76-77 行：指针偏移计算
X += row * stride_x_row + group * N
Y += row * stride_y_row + group * N
```

**说明**：
- `row * stride_x_row`：跳转到第 `row` 行的起始位置
- `group * N`：跳转到第 `group` 组的起始位置
- 对于连续存储的张量，`stride_x_row = N`

### 数据访问模式

```
行 0:  [g0: 0→127] [g1: 128→255] [g2: 256→383] [g3: 384→511]
行 1:  [g0: 0→127] [g1: 128→255] [g2: 256→383] [g3: 384→511]
...
行 M-1:[g0: 0→127] [g1: 128→255] [g2: 256→383] [g3: 384→511]
```

每个 `(row, group)` 程序独立处理一个块。

---

## 形状变化流程图

```
用户输入:
  x.shape = [batch, seq_len, hidden_dim]
        ↓
保存原始形状:
  x_shape_og = [batch, seq_len, hidden_dim]
        ↓
重塑为 2D:
  x.shape = [M, N]
  M = batch * seq_len
  N = hidden_dim
        ↓
创建输出:
  out = torch.empty_like(x)
  out.shape = [M, N]
        ↓
Triton Kernel 执行:
  grid = (M, ngroups)
  每个程序: 处理 [row, group*group_size : (group+1)*group_size]
        ↓
Kernel 输出:
  out.shape = [M, N]  (填充完毕)
        ↓
恢复原始形状:
  out.shape = [batch, seq_len, hidden_dim]
        ↓
返回用户:
  result.shape = [batch, seq_len, hidden_dim]
```

---

## 与其他归一化的对比

| 归一化类型 | 输入形状 | 输出形状 | 特点 |
|-----------|----------|----------|------|
| LayerNorm | `[..., N]` | `[..., N]` | 形状不变 |
| BatchNorm | `[B, C, ...]` | `[B, C, ...]` | 形状不变 |
| GroupNorm | `[..., N]` | `[..., N]` | 形状不变 |
| InstanceNorm | `[B, C, ...]` | `[B, C, ...]` | 形状不变 |

**共同点**：所有归一化操作都保持输入输出形状一致。

---

## 代码验证

### 验证脚本

```python
import torch
from sglang.srt.layers.attention.fla.layernorm_gated import layernorm_fn

# 测试不同形状的输入
test_cases = [
    (8, 128, 768),           # [batch, seq_len, hidden_dim]
    (4, 64, 32, 512),        # [batch, height, width, channels]
    (1024, 2048),            # [tokens, hidden_dim]
    (2, 3, 4, 5, 128),       # 5D tensor
]

for shape in test_cases:
    x = torch.randn(*shape, device='cuda')
    N = shape[-1]
    weight = torch.ones(N, device='cuda')
    bias = torch.zeros(N, device='cuda')
    
    # 调用 LayerNorm
    out = layernorm_fn(x, weight, bias)
    
    # 验证输出形状
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"✓ Input: {x.shape} → Output: {out.shape}")
```

### 预期输出

```
✓ Input: torch.Size([8, 128, 768]) → Output: torch.Size([8, 128, 768])
✓ Input: torch.Size([4, 64, 32, 512]) → Output: torch.Size([4, 64, 32, 512])
✓ Input: torch.Size([1024, 2048]) → Output: torch.Size([1024, 2048])
✓ Input: torch.Size([2, 3, 4, 5, 128]) → Output: torch.Size([2, 3, 4, 5, 128])
```

---

## 总结

### 🎯 核心答案

**`_layer_norm_fwd_1pass_kernel` 的输出 `out` 的形状为 `[M, N]`**

其中：
- `M`：所有前导维度的乘积（行数）
- `N`：最后一个维度（特征维度）

**最终返回给用户的输出形状与输入形状完全一致。**

### 关键要点

1. ✅ **形状不变性**：输出形状始终等于输入形状
2. ✅ **2D 处理**：内部将多维输入重塑为 2D 进行高效处理
3. ✅ **形状恢复**：处理完成后恢复到原始输入形状
4. ✅ **特征维度保持**：最后一个维度（特征维度）不变
5. ✅ **支持任意维度**：支持 2D、3D、4D 或更高维度的输入

### 设计优势

- **通用性**：支持任意形状的输入张量
- **高效性**：内部统一为 2D 处理，利用 Triton 优化
- **灵活性**：支持标准 LayerNorm 和 GroupNorm
- **可组合性**：支持可选的 gating 机制（与 z 分支组合）

---

最后更新: 2025-01-17

