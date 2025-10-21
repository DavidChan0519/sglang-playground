# test_cu_seqlens.py 重构总结

## ✅ 完成时间: 2025-10-20

---

## 🎯 重构目标

用户要求重构 `test_cu_seqlens.py`，使 device 可以统一配置，类似于 `test_fused_sigmoid_gating_native_refactored.py` 的实现方式。

---

## 🔧 主要改进

### 1. **添加命令行参数支持**

```python
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
```

### 2. **全局 device 配置**

**重构前**:
```python
def test_varlen_vs_fixed_length():
    device = 'cpu'  # 硬编码
    # ...
    A_log = torch.randn(HV, dtype=dtype, device=device)
```

**重构后**:
```python
# 全局变量
DEVICE = None  # 将在 parse_args 中设置

def test_varlen_vs_fixed_length():
    # ...
    A_log = torch.randn(HV, dtype=dtype, device=DEVICE)  # 使用全局变量
```

### 3. **统一的 Triton 对比控制**

```python
# 在测试函数中
if HAS_TRITON and not SKIP_TRITON:
    # 运行 Triton 对比
    print("\n[4] Triton 实现（变长模式）")
    # ...
elif not HAS_TRITON:
    print(f"\n⚠️  跳过 Triton 对比: {TRITON_IMPORT_ERROR}")
elif SKIP_TRITON:
    print(f"\n⏭️  跳过 Triton 对比 (--skip-triton)")
```

### 4. **自动 CUDA 检测和回退**

```python
if DEVICE.startswith('cuda'):
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print(f"❌ CUDA 不可用，将回退到 CPU")
        DEVICE = 'cpu'
```

### 5. **重构主函数结构**

```python
def run_all_tests():
    """运行所有测试"""
    global DEVICE
    
    # 解析命令行参数
    args = parse_args()
    
    # 显示配置
    # ...
    
    # 运行测试
    results = []
    # ...
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

---

## 📋 使用方法

### 基本用法

#### 1. CPU 测试（默认）
```bash
python3 test_cu_seqlens.py
# 或
python3 test_cu_seqlens.py --device cpu
```

#### 2. CUDA 测试
```bash
python3 test_cu_seqlens.py --device cuda
```

#### 3. 指定 CUDA 设备
```bash
python3 test_cu_seqlens.py --device cuda:0
python3 test_cu_seqlens.py --device cuda:1
```

#### 4. 跳过 Triton 对比
```bash
python3 test_cu_seqlens.py --device cuda --skip-triton
```

#### 5. 查看帮助
```bash
python3 test_cu_seqlens.py --help
```

输出:
```
usage: test_cu_seqlens.py [-h] [--device DEVICE] [--skip-triton]

测试 cu_seqlens (变长序列) 功能

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  测试设备 (cpu, cuda, cuda:0, etc.). 默认: cpu
  --skip-triton    跳过 Triton 对比测试
```

---

## ✅ 测试验证

### CPU 测试
```bash
$ python3 test_cu_seqlens.py --device cpu

╔==============================================================================╗
║                    cu_seqlens 变长序列测试                            ║
╚==============================================================================╝

✅ PyTorch 版本: 2.3.0+cpu
✅ 测试设备: cpu
✅ Triton 实现: 不可用
   原因: No module named 'pybase64'

================================================================================
测试 1: 变长序列 vs 固定长度
================================================================================
配置: N=3, seq_lens=[5, 7, 6], total_len=18, device=cpu
       H=2, HV=2, K=8, V=8
       cu_seqlens=[0, 5, 12, 18]

[1] Native 实现（变长模式）
输出: shape=torch.Size([1, 18, 2, 8]), mean=-0.000037

[2] Optimized 实现（变长模式）
输出: shape=torch.Size([1, 18, 2, 8]), mean=-0.000037

✅ Native vs Optimized (变长模式): 最大差异 2.33e-10

[3] 固定长度模式（逐序列对比）
  序列 0: 长度=5, 范围=[0:5]
    Fixed vs Varlen: 最大差异 0.00e+00
    ✅ 一致
  序列 1: 长度=7, 范围=[5:12]
    Fixed vs Varlen: 最大差异 0.00e+00
    ✅ 一致
  序列 2: 长度=6, 范围=[12:18]
    Fixed vs Varlen: 最大差异 0.00e+00
    ✅ 一致

⚠️  跳过 Triton 对比: No module named 'pybase64'

================================================================================
测试 2: 变长序列 + 初始状态
================================================================================
配置: N=3, seq_lens=[4, 6, 5], num_states=5, device=cpu
Native 输出: shape=torch.Size([1, 15, 2, 8]), mean=-0.000121
最终状态[0]: mean=0.000374
最终状态[2]: mean=0.000515
最终状态[1]: mean=0.000060
Optimized 输出: shape=torch.Size([1, 15, 2, 8]), mean=-0.000121

✅ Native vs Optimized
   输出: 最大差异 9.31e-10
   最终状态: 最大差异 4.66e-10

================================================================================
测试总结
================================================================================
变长 vs 固定长度                    : ✅ 通过
变长 + 初始状态                     : ✅ 通过

🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 
✅ 所有 cu_seqlens 测试通过！
🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 
```

### 使用 --skip-triton 参数
```bash
$ python3 test_cu_seqlens.py --device cpu --skip-triton

✅ PyTorch 版本: 2.3.0+cpu
✅ 测试设备: cpu
✅ Triton 实现: 不可用
   原因: No module named 'pybase64'
⏭️  跳过 Triton 对比测试 (--skip-triton)

# ... 测试运行 ...
```

---

## 🔄 与重构前的对比

| 特性 | 重构前 | 重构后 |
|------|--------|--------|
| Device 配置 | 硬编码 `device = 'cpu'` | 命令行参数 `--device` ✅ |
| Triton 对比控制 | 自动（基于可用性） | 可控制 `--skip-triton` ✅ |
| CUDA 检测 | 无 | 自动检测并回退 ✅ |
| 命令行帮助 | 无 | `--help` 参数 ✅ |
| 灵活性 | 低（需修改代码） | 高（命令行配置）✅ |
| 可维护性 | 中等 | 高 ✅ |
| 代码一致性 | 与其他测试不一致 | 与 refactored 版本一致 ✅ |

---

## 📂 修改的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `test_cu_seqlens.py` | 重构 | 添加命令行参数支持 |

---

## 🎓 关键改进点

### 1. **代码一致性**
现在 `test_cu_seqlens.py` 与 `test_fused_sigmoid_gating_native_refactored.py` 使用相同的结构和模式，提高了代码库的一致性。

### 2. **灵活性**
用户可以轻松切换测试设备，无需修改代码：
```bash
# 快速切换设备
python3 test_cu_seqlens.py --device cpu
python3 test_cu_seqlens.py --device cuda
python3 test_cu_seqlens.py --device cuda:1
```

### 3. **可扩展性**
如果未来需要添加更多配置选项（如 `dtype`、测试规模等），只需在 `parse_args()` 中添加新参数即可。

### 4. **错误处理**
自动检测 CUDA 可用性并回退到 CPU，避免运行时错误。

---

## 🔍 技术细节

### 全局变量管理

```python
# 模块级别
DEVICE = None
SKIP_TRITON = False

# 在 parse_args() 中设置
def parse_args():
    global DEVICE, SKIP_TRITON
    DEVICE = args.device
    SKIP_TRITON = args.skip_triton

# 在 run_all_tests() 中可能修改（CUDA 回退）
def run_all_tests():
    global DEVICE
    if DEVICE.startswith('cuda') and not torch.cuda.is_available():
        DEVICE = 'cpu'
```

### 测试函数访问

所有测试函数直接使用全局变量 `DEVICE`，无需传参：

```python
def test_varlen_vs_fixed_length():
    # 直接使用全局 DEVICE
    cu_seqlens = torch.tensor(..., device=DEVICE)
    A_log = torch.randn(..., device=DEVICE)
```

---

## ✅ 完成检查清单

- ✅ 添加命令行参数解析
- ✅ 全局 DEVICE 变量
- ✅ 全局 SKIP_TRITON 变量
- ✅ 更新测试函数 1（变长 vs 固定长度）
- ✅ 更新测试函数 2（变长 + 初始状态）
- ✅ 添加 CUDA 检测和回退
- ✅ 重构主函数结构
- ✅ 测试 CPU 模式
- ✅ 测试 --help 参数
- ✅ 测试 --skip-triton 参数
- ✅ 所有测试通过

---

## 📝 总结

### ✅ 重构成功

`test_cu_seqlens.py` 现在：
1. **支持命令行参数** - `--device` 和 `--skip-triton`
2. **代码结构一致** - 与 `test_fused_sigmoid_gating_native_refactored.py` 保持一致
3. **灵活易用** - 无需修改代码即可切换测试配置
4. **鲁棒性强** - 自动检测 CUDA 可用性并回退
5. **所有测试通过** - 功能完全正常

### 🎯 使用建议

- **日常开发**: 使用 `--device cpu` 快速测试
- **CI/CD**: 根据环境自动选择设备
- **调试**: 使用 `--skip-triton` 专注 Native 实现
- **性能测试**: 使用 `--device cuda` 进行 GPU 测试

---

**重构完成时间**: 2025-10-20  
**测试状态**: ✅ 全部通过  
**代码质量**: ✅ 高

