
# 对 prefill only models 推理性能调优 

## 引言
本文关注以下模型，比如：
- xlm_roberta
- bge-m3
- bge-reranker-v2-m3
- bert
- bge v1.5 family
- Snowflake Arctic Embed (Family)
- gte-Qwen2
- ...

这些模型大致可以分为三类：
- Encode only models. (Bidirectional Transformers, causal=False)，经常微调为retriever、reranker。。。
- Decode only models. (masked multi-head attention, causal=True)，有以下两个有意思的用途
  - 作为特征提取器，输出最后的隐藏状态 (Output last hidden states
  - Decode only retriever (比如 e5-mistral-7b
  - 是否经过微调，对于推理几乎没有任何区别
- Enable bidirectional. [LLM2Vec](https://arxiv.org/abs/2404.05961) 提出了一种简单的无监督方法，可以将任何 Decode only models 转换为强大的 Encode only models。

以上三类的共同点是都只有预填充阶段，prefill only。

为了使术语更加精确，下面使用prefill only。你可以将下文的 prefill only 替换为 encode only 方便理解。

## 如何评估推理性能
我一般会通过依次采样多个不同的batchsize，得到相应的吞吐和响应，绘制吞吐延迟曲线评估推理性能

横坐标为单位qps，纵坐标为延迟单位毫秒。延迟低吞吐高表明模型推理性能好，所以吞吐延迟曲线评估右下更好

> 吞吐量（Throughput）定义和计算比较明确，单位时间里完成的请求
> 
> 延迟 （Latency）计算口径比较多，下面每张图每个表的口径都可能不太一样，要注意区分


> 本文以 bge-m3 模型在单张 4090 推理性能举例

```commandline
python -m benchmarks.retriever.benchmark_attention_impl
```

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/imps/fp32-sync.png?raw=true" width="400">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/imps/fp16-sync.png?raw=true" width="400">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/imps/bf16-sync.png?raw=true" width="400">

> 以上三张图的延迟计算口径为模型执行时间，反应模型推理性能，包括数据H2D，GPU计算，结果D2H。不包括预处理、调度、后处理。

可以看到
- 对模型推理性能影响最大的是浮点数据格式，bf16 ≈ fp16 > fp32
- 其次attention的实现对理性能影响也很大， FLASH_ATTN = FLASHINFER > XFORMERS > TORCH_SDPA > TORCH_NAIVE
> 在 prefill only models 用Flashinfer时，实际上使用的是FLASH ATTN

再看与 transformers 库的对比

```commandline
python -m benchmarks.retriever.benchmark_bge-m3
```

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/imps/fp16-hf.png?raw=true" width="400">

> 上图延迟计算口径为离线批处理端到端时间，包括 tokenizer.encode 时间，GPU计算时间，transformers 没有执行结果D2H， wde框架执行结果D2H，当然差别不大


> [RoBERTa-based Add support for sdpa #30510](https://github.com/huggingface/transformers/pull/30510)

- v4.45.0 之前没有用 sdpa 优化，吞吐延迟曲线跟TORCH_NAIVE差不多
- v4.45.0 使用 sdpa 优化，性能有比较大的提升，batchsize小的时候，FLASH_ATTN 还能拉开一些差距， batchsize 大的时候甚至可以稍微赶超FLASH_ATTN。 

sdpa 优化的 transformers 已经不好欺负了， 作为这次推理性能调优的旅程的起点


> 本文以下使用 bge-m3 模型在单张 4090 使用 FLASH_ATTN 和 fp16 推理性能举例

## 调度优化

先看图，经过同步调度 sync、简单异步调度 simple_async、异步调度 async、双缓存 double_buffer优化之后，性能明显提升

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/execute.png?raw=true" width="400">

> 上图的延迟计算口径为模型执行时间，反应模型推理性能，包括数据H2D，GPU计算，结果D2H。不包括预处理、调度、后处理。

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/e2e.png?raw=true" width="400">

> 上图延迟计算口径为离线批处理端到端时间，包括 tokenizer.encode，GPU计算，结果D2H，后处理

以上两张图被称为 “离线不同调度吞吐-延迟图”


### 同步调度分析
```commandline
python -m benchmarks.profiler.profiling_executor
```

[同步调度代码](https://github.com/noooop/wde/blob/c1920124c750ea2c66bce0b28ad443e615467995/wde/tasks/prefill_only/executor/gpu_executor.py#L127C1-L143C30)

使用 chrome://tracing/ 查看 sync_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/sync.png?raw=true" width="400">

可以看到同步调度时，两次模型计算之间有一些空隙，系统在运行上一个批次请求的后处理和下一个批次调度、预处理，GPU处于空闲状态。

从 “离线不同调度吞吐-延迟图” sync曲线可以看到，同步调度性能跟transformers差不多。

如何能消除GPU空闲，提高GPU利用率，提高系统吞吐呢？

### 简单异步调度分析

[简单异步调度代码](https://github.com/noooop/wde/blob/c1920124c750ea2c66bce0b28ad443e615467995/wde/tasks/prefill_only/executor/gpu_executor.py#L145C1-L157C32)

将系统改成异步调度
- scheduler 和 executor 使用 queue 传输输入输出
- scheduler 将 batch 放入 input_queue，不等上一个 batch 返回，立即调度下一个 batch。
- input_queue 总是有多个 batch 供 executor 使用
- executor 执行完上一个 batch，将结果放入 output_queue，立即执行下一个 batch

使用 chrome://tracing/ 查看 simple_async_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/simple_async.png?raw=true" width="400">

可以看到简单异步调度，基本上消除了GPU空闲

从 “离线不同调度吞吐-延迟图” 可以看到simple_async曲线跟之前sync曲线，有一定提高

接下来能如何提高系统性能呢?


### non_blocking

参考pytorch官方的 [non_blocking 教程](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)， 使用多个 cuda.Stream 结合 non_blocking 可以加速系统运行

直接跳到结论，通过多个cuda.Stream并行计算，不仅io和计算并行可以提高性能，两个batch计算并行也可以提高性能 

[non_blocking 异步调度代码](https://github.com/noooop/wde/blob/c1920124c750ea2c66bce0b28ad443e615467995/wde/tasks/prefill_only/executor/gpu_executor.py#L159C1-L190C26)

使用 chrome://tracing/ 查看 async_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/async.png?raw=true" width="400">

从 “离线不同调度吞吐-延迟图” 可以看到 async 曲线，相对simple_async曲线、sync曲线，效果非常显著

### double_buffer

上面 non_blocking async 调度，本质上是一种使用了两个cuda.Stream、两个batch，两个batch计算并行的调度方式

既然batch之间计算并行可以提高性能，那就设计一种使用两个cuda.Stream、三个batch，两个batch计算并行的调度方式

[double_buffer 异步调度代码](https://github.com/noooop/wde/blob/c1920124c750ea2c66bce0b28ad443e615467995/wde/tasks/prefill_only/executor/gpu_executor.py#L192C1-L282C26)

使用 chrome://tracing/ 查看 double_buffer_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/profiler/double_buffer.png?raw=true" width="400">

虽然 double_buffer 看起来很酷，但从 “离线不同调度吞吐-延迟图” 可以看到 double_buffer 对比 async 几乎没有性能提升

所以虽然可以设计一个使用 100 个 cuda.Stream、1000 个 batch，10000 个 batch计算并行的调度方式，但估计也不会更多的有性能提升，甚至可能有性能下降。

尤其是在线服务，很难同时凑满 double_buffer 使用的三个 batch，所以 async 调度为本系统默认调度方式

## 未完待续