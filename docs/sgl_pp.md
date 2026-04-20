# SGLang 长上下文流水线并行(Pipeline Parallelism)总结

> 原文链接:<https://docs.sglang.io/advanced_features/pipeline_parallelism.html>

这篇文档介绍了 SGLang 针对超长上下文推理场景引入的 **Pipeline Parallelism (PP)** 能力,以及配套的 **Dynamic Chunked Prefill** 机制。

---

## 一、为什么需要 PP?

当模型向万亿参数和超长上下文演进时,TTFT(Time to First Token)成为主要瓶颈:

- KV cache 只能减少冗余计算,**无法绕开超长 ITL(Input Token Length) 带来的高 TTFT**
- TP 在多节点部署时会遇到通信瓶颈;而 PP **只需在每个 stage 边界做跨节点通信**,计算-通信 overlap 更好

**核心收益**:同一个请求的输入 token 被切成多个 chunk,不同 chunk 可以在不同节点上并行处理,从而降低 TTFT。

---

## 二、基于异步通信的实现重构

SGLang 早期版本已支持 PP(#5724)并兼容 PD 分离(#8846),但性能不理想。新版实现(#11852)的关键机制如下。

### 1. Micro-batching Event Loop + 非阻塞 P2P 通信

- Scheduler 使用 `async_send` 返回 `P2PWork` handle,**不阻塞等待传输完成**
- 真正的同步 `P2PWork.work.wait()` 延后到 `_pp_commit_comm_work`
- 这样在数据传输的同时,CPU 可以去调度下一个 batch 或处理 metadata

### 2. 多流执行

| Stream | 作用 |
|---|---|
| `default_stream` | 同步主流 |
| `forward_stream` | 前向计算 |
| `copy_stream` | D2H 内存拷贝 |

当前 micro-batch 在 GPU 计算时,CPU 并行处理上一个 micro-batch 的结果 (`_pp_process_batch_result`),从而保持流水线尽可能饱和。

---

## 三、Dynamic Chunking(动态分块)

### 3.1 为什么需要动态分块

固定 chunk size 会导致流水线 bubble,尤其当 PP size 较大时。根本原因:

> **Transformer 结构下,即使 chunk size 相同,prefix 越长单 chunk 运行时间越长。**

这种不均衡会在 stage 之间传播,严重降低大 PP 的扩展效率。

### 3.2 核心公式

动态 chunking 希望满足:

```
Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)
```

其中 `L` 是 prefix 长度。

- 对不同 ITL 做性能 profiling
- 把累积 runtime 建模成序列长度的**二次函数**
- 反解出下一块最优 chunk size

由于 Attention 复杂度随 `L` 增长,实际上 **chunk size 会随 L 增大逐步缩小**,以保持每一段执行时间对齐。

最终预测值会**向下对齐**到 `max(--page-size, 64)` 的倍数,以便 KV cache 管理和硬件对齐。

### 3.3 关键参数

| 参数 | 作用 |
|---|---|
| `--enable-dynamic-chunking` | 开启动态分块 |
| `--chunked-prefill-size` | 开启动态分块后作为**初始** chunk size,建议设大一些 |
| `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` | 平滑因子,默认 `0.75` |

**平滑因子的含义**:

| 值 | 含义 |
|---|---|
| `1.0` | 严格遵循二次模型预测 |
| `0.6 – 0.85` | **推荐区间**,兼顾动态调整和硬件稳定性 |
| `0` | 关闭动态调整,等价于固定 chunk size |

---

## 四、长上下文最佳实践(调优三步法)

### Step 1 — 找到固定 chunk 的最优 baseline

从 4K 开始,逐步尝试至 16K,针对目标 PP size 和 ITL 找到最优固定 chunked prefill size。

### Step 2 — 动态分块的初始 chunk size 选择

- 设为固定最优值的 **2× 或 3×**
- 极长 ITL 场景可用 **4×**
- 动态预测器会保证后续 chunk 不小于初始值的 `1/4`,避免"尾部 chunk"浪费算力

### Step 3 — 平滑因子调优

在 `0.6 – 0.85` 之间微调。

### 额外优化技巧 —— 不均分层时的分区策略

当层数无法均匀划分到各 PP rank 时,**把更大的分区放到更高的 PP rank**。

例如 DeepSeek-V3.1:

```bash
SGLANG_PP_LAYER_PARTITION=15,15,15,16   # ✅ 推荐
SGLANG_PP_LAYER_PARTITION=16,15,15,15   # ❌ 不推荐
```

原因:高 rank 等待前一级结果时可利用更多计算时间,减少 bubble。

---

## 五、H20 上的案例(128K ITL,4 节点)

| 模型 | 并行配置 | 最优固定 chunk | 动态 chunking 初始 size | smooth factor |
|---|---|---|---|---|
| DeepSeek-V3.1 | `tp=8, pp=4` | 4K | 12K (3×) | **0.65** |
| Qwen3-235B-A22B-FP8 | `tp=4, pp=8` | 6K | 18K (3×) | **0.8** |

### 启动命令示例 — DeepSeek-V3.1(固定 chunk)

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8 \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096
```

### 启动命令示例 — DeepSeek-V3.1(动态 chunking)

```bash
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8 \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

### 启动命令示例 — Qwen3-235B-A22B-FP8(动态 chunking)

```bash
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8 \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

> ⚠️ `--disable-radix-cache` 仅用于可复现 benchmark,**生产环境不建议**。

---

## 六、一句话总结

> SGLang 的 PP 方案通过「**异步 P2P + 多流**」实现计算-通信 overlap,再配合「**基于二次模型的动态 chunking**」解决 Transformer prefill 阶段 chunk 耗时不均导致的 pipeline bubble,从而在长上下文场景下降低 TTFT。
>
> 关键调优手段:
>
> 1. **初始 chunk size = 固定最优 × 2~3**
> 2. **smooth factor 在 0.6–0.85 之间扫参**
> 3. **层分区向高 PP rank 倾斜**