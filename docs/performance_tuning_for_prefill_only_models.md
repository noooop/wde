
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

> 图1 以上三张图的延迟计算口径为模型执行时间，反应模型推理性能，包括数据H2D，GPU计算，结果D2H。不包括预处理、调度、后处理。

可以看到
- 对模型推理性能影响最大的是浮点数据格式，bf16 ≈ fp16 > fp32
- 其次attention的实现对理性能影响也很大， FLASH_ATTN = FLASHINFER > XFORMERS > TORCH_SDPA > TORCH_NAIVE
> 在 prefill only models 用Flashinfer时，实际上使用的是FLASH ATTN

再看与 transformers 库的对比

```commandline
python -m benchmarks.retriever.benchmark_bge-m3
```

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/imps/fp16-hf.png?raw=true" width="400">

> 图2 上图延迟计算口径为离线批处理端到端时间，包括 tokenizer.encode 时间，GPU计算时间，transformers 没有执行结果D2H， wde框架执行结果D2H，当然差别不大


> [RoBERTa-based Add support for sdpa #30510](https://github.com/huggingface/transformers/pull/30510)

- v4.45.0 之前没有用 sdpa 优化，吞吐延迟曲线跟TORCH_NAIVE差不多
- v4.45.0 使用 sdpa 优化，性能有比较大的提升，batchsize小的时候，FLASH_ATTN 还能拉开一些差距， batchsize 大的时候甚至可以稍微赶超FLASH_ATTN。 

sdpa 优化的 transformers 已经不好欺负了， 作为这次推理性能调优的旅程的起点


> 本文以下使用 bge-m3 模型在单张 4090 使用 FLASH_ATTN 和 fp16 推理性能举例

## 离线批量推理优化性能

为了更好的对系统性能观察和调优，记录了以下Metrics。一个请求的生命周期如下：

时间戳
- arrival_ts：请求进入系统时间戳
- scheduled_ts：调度器调度请求的时间戳
- inference_begin_ts：执行器执行开始的时间戳
- inference_end_ts：执行器执行完成的时间戳
- finish_ts：请求完成的时间戳

通过时间戳可以计算以下时间
- waiting_time = scheduled_ts - arrival_ts： 请求在调度前队列里等待的时间
- scheduling2inference = inference_begin_ts - scheduled_ts：请求从调度到执行的时间
- inference_time = inference_end_ts - inference_begin_ts：请求执行器推理的时间
- latency = finish_ts - scheduled_ts：请求从调度到完成的时间

离线批量推理情景下 waiting_time 跟总请求量有关，一般不关注，所以有两个延迟口径需要关注：
- inference_time = inference_end_ts - inference_begin_ts：请求执行器推理的时间
- latency = finish_ts - scheduled_ts：请求从调度到完成的时间

先看图，经过同步调度 sync、简单异步调度 simple_async、异步调度 async 优化之后，性能明显提升

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/inference_time.png?raw=true" width="400">

> 图3 上图的延迟计算口径为 inference_time = inference_end_ts - inference_begin_ts，请求执行器推理的时间

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/latency.png?raw=true" width="400">

> 图4 上图延迟计算口径为 latency = finish_ts - scheduled_ts，请求从调度到完成的时间


### 同步调度分析
```commandline
python -m benchmarks.retriever.profiler.profiling_executor
```

[同步调度代码](https://github.com/noooop/wde/blob/823ea72d43b2c8cdbfb6a55e65010e18082feb4d/wde/workflows/core/executor/gpu_executor.py#L125C1-L153C30)

使用 chrome://tracing/ 查看 sync-1-1.json ~ sync-1-64.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/sync-1.png?raw=true" width="400">
<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/sync-2.png?raw=true" width="400">

> 图5、图6 对于轻负载, 推理瓶颈为 cuda kernels launch 的速度

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/sync-3.png?raw=true" width="400">

> 图7 对于重负载，两次模型计算之间有一些空隙，系统在运行上一个批次请求的后处理和下一个批次调度、预处理，GPU处于空闲状态。

尤其是每个请求都需要复杂预处理后处理的任务，GPU处于空闲状态的比例不容忽视。

从 图3图4 sync曲线可以看到，同步调度性能跟transformers差不多。

如何能消除GPU空闲，提高GPU利用率，提高系统吞吐呢？

### 简单异步调度分析

[简单异步调度代码](https://github.com/noooop/wde/blob/823ea72d43b2c8cdbfb6a55e65010e18082feb4d/wde/workflows/core/executor/gpu_executor.py#L155C1-L167C32)

将系统改成异步调度
- scheduler 和 executor 使用 queue 传输输入输出
- scheduler 将 batch 放入 input_queue，不等上一个 batch 返回，立即调度下一个 batch。
- input_queue 总是有多个 batch 供 executor 使用
- executor 执行完上一个 batch，将结果放入 output_queue，立即执行下一个 batch

使用 chrome://tracing/ 查看 sync-1-1.json ~ sync-1-64.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/simple_async-1.png?raw=true" width="400">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/simple_async-2.png?raw=true" width="400">

> 图8、 图9  可以看到简单异步调度，基本上消除了两次模型计算之间有一些空隙，对于重负载 GPU 利用率得到提高

从 图3图4 可以看到simple_async曲线跟之前sync曲线，峰值QPS有显著提升，但是batchsize=1几乎没有提升

异步调度本质是两个或多个batch交替在GPU上执行，所以单个请求的延迟基本上要翻翻，从图9就可以看出来。

好处是只要CPU上花的时间小于GPU花的时间，完全可以覆盖掉，这种优化方式在CUDA Parallel Programming 称为 Tiling.

接下来能如何提高系统性能呢?

### non_blocking

参考pytorch官方的 [non_blocking 教程](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)， 使用多个 cuda.Stream 结合 non_blocking 可以加速系统运行

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/async-1.png?raw=true" width="400">

使用 chrome://tracing/ 查看 async-1-1.json ~ async-1-64.json


> 图10 可以看到 io 和 计算确实可以并行

从 图3图4 可以看到 async-1 曲线几乎和 simple_async曲线贴在一起， io 化的时间占比很小，所以几乎起不到优化作用

### non_blocking +

直接跳到结论，通过多个cuda.Stream并行计算，不仅io和计算并行可以提高性能，两个batch计算并行也可以提高性能 

这时候我们需要两个线程，async-N 中的 N 表示 有几个计算线程

使用 chrome://tracing/ 查看 查看 async-2-1.json ~ async-2-64.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/async-2-1.png?raw=true" width="400">

> 图11 对于轻负载，两个线程 cuda kernels launch 肯定比一个效率高

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/async-2-2.png?raw=true" width="400">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/retriever/async-2-3.png?raw=true" width="400">

> 图12、图13 对于重负载，算子之间都可以并行，进一步提高了GPU利用率

从 图3图4 可以看到 async-2 曲线，对于batchsize=1，qps提升很大，对于重负载也有一定提升

### non_blocking ++

上面 non_blocking async 调度，本质上是一种使用了两个cuda.Stream、两个batch，两个线程并行的异步调度方式

既然batch之间计算并行可以提高性能，那就设计一种使用三个cuda.Stream、三个batch，三个线程并行的异步调度方式

从 图3图4 可以看到 async-3 曲线，几乎没有提升，所以两个线程并行的异步调度方式已经可以让GPU饱和

## 未完待续