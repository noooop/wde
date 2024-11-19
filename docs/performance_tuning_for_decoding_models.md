
# 对 Decoding (chat) models 推理性能调优 

安利姐妹篇 [对 prefill only models 推理性能调优](https://github.com/noooop/wde/blob/main/docs/performance_tuning_for_prefill_only_models.md)

## 引言
1. 首字延迟 & 生成延迟
- Decoding (chat) models 推理过程可以分成，预填充 (Prefill) 和解码 （Decoding） 两个阶段
- 预填充 (Prefill) 阶段对应于输出第一个词的的，需要将提示词每个词的中间结果都写入 kv cache 里，才能推理得到第一个词。 这个阶段是对 kv cache 进行预填充。
- 首字延迟（First Token Latency、Time To First Token (TTFT)）也就是系统生成第一个字符所需的响应时间
- 解码 （Decoding） 阶段对应于输出之后的 token 的输出。 kv cache 已经填入之前 token。 推理过程就是从下往上过一遍模型，最后从 lm_head 输出 logits 采样。
- 生成延迟（Decoding Latency、Time Per Output Token (TPOT)）之后的字符响应时间。

2. 推理速度极限
- 现代 CPU/GPU 的 ALU 操作（乘法、加法）内存 IO 速度要快得多，transformer 这种只需要对每个元素执行两次操作的场景，必定受到访存带宽的限制。
- 以 NVIDIA GeForce RTX 4090 为例，1008 GB/s 显存带宽和 83 TFLOPS，Flop:byte = 82:1，算力非常充足。
- 对于解码 （Decoding） 阶段，每输出一个词需要读一遍模型，读取模型的极限就是速度的极限
- 对于预填充 (Prefill) 阶段，读一遍模型计算提示词中的多个词，无论从 SIMD （Single Instruction Multiple Data） 的角度还是从带宽瓶颈 vs 算力瓶颈的角度都很划算
- 更多推理速度极限细节情参考 [LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/)

3. 显存管理
- LLM 输出总长度事先未知，kv cache 需要按需不断增加，为 llm 推理框架增加了不少麻烦。简单的显存管理会产生显存碎片（fragmentation）
- 受到操作系统的虚拟内存分页启发，vLLM 引入了 PagedAttention 减轻 kv cache 的显存碎片。
- 更多显存管理细节情参考
  - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
  - [vAttention](https://arxiv.org/abs/2405.04437)

4. 大语言模型多用户场景下性能优化技术
- Continuous batching
  - LLM 输出总长度不统一，输出短的请求会提前退出，资源空闲，影响整体吞吐率。连续批处理(Continuous batching)，当有请求退出，资源空闲时，不等所有请求都完成，中途及时将等待队列中的请求加入到批次中，减少流水线气泡，提高资源利用率，提高整体吞吐率。
  - 连续批处理(Continuous batching)的关键是一次模型推理，同时进行解码 (Decoding) 阶段和预填充 (Prefill) 阶段。
  - 下图来自 [continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
  - <img src="https://images.ctfassets.net/xjan103pcp94/744TAv4dJIQqeHcEaz5lko/b823cc2d92bbb0d82eb252901e1dce6d/cb_03_diagram-continuous-batching.png" width="800">
  - 可以看到使用连续批处理(Continuous batching) 技术，流水线被填充的满满当当，非常好。
  - 更多细节请参考 [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)
- Dynamic SplitFuse / Chunked Fill 
  - 如果一个请求的预填充 (Prefill) 阶段在一次模型中完成，超大的请求会导致解码阶段的用户也要跟着等非常长时间，显然很不合理
  - 下图来自 [DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/chinese)
  - <img src="https://github.com/microsoft/DeepSpeed/raw/master/blogs/deepspeed-fastgen/assets/images/fastgen-overview-light.png" width="800">
  - 更多Dynamic SplitFuse / Chunked Fill 细节请参考：
    - [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/abs/2401.08671)
    - [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
    - [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
    - [vllm Chunked Prefill](https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill)

## 如何评估推理性能
wde 使用 Chunked Fill，所以不区分预填充 (Prefill) 和解码 (Decoding)阶段，也就是一个step既有Prefill阶段的任务，也有Decoding阶段的任务。

一个step的执行时间，基本上只与  max_num_batched_tokens 或者叫 chunked_prefill_size 有关， 具体细节参考 [DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671)

我一般会通过依次采样多个不同的 max_num_batched_tokens，得到相应的吞吐和响应，绘制吞吐延迟曲线评估推理性能


横坐标为单位qps，纵坐标为延迟单位毫秒。延迟低吞吐高表明模型推理性能好，所以吞吐延迟曲线评估右下更好

> 吞吐量（Throughput）定义和计算比较明确，单位时间里完成的请求
> 
> 延迟 （Latency）计算口径比较多，下面每张图每个表的口径都可能不太一样，要注意区分

> 本文以 Qwen2.5-7B-Instruct fp8模型在单张 4090 推理性能举例

## 离线批量推理优化性能

为了更好的对系统性能观察和调优，记录了以下Metrics。一个请求会输出多个token，也就是一个请求会解码多次，请求的一次解码生命周期如下：

时间戳
- arrival_ts：进入系统时间戳
- scheduled_ts：调度器调度时间戳
- first_scheduled_ts：请求第一次被调度的时间戳
- inference_begin_ts：执行器执行开始的时间戳
- inference_end_ts：执行器执行完成的时间戳
- finish_ts：完成的时间戳

通过时间戳可以计算以下时间
- waiting_time = arrival_ts - first_scheduled_ts： 请求在调度前队列里等待的时间
- scheduling2inference = inference_begin_ts - scheduled_ts：从调度到执行的时间
- inference_time = inference_end_ts - inference_begin_ts：执行器推理的时间
- latency = finish_ts - scheduled_ts：从调度到完成的时间

离线批量推理情景下 waiting_time 跟总请求量有关，一般不关注，所以有两个延迟口径需要关注：
- inference_time = inference_end_ts - inference_begin_ts：执行器推理的时间
- latency = finish_ts - scheduled_ts：从调度到完成的时间

### 同步调度分析
```commandline
python -m benchmarks.chat.profiler.profiling_decoding
```

[同步调度代码](https://github.com/noooop/wde/blob/fb361f890bc2128175304d9788312be75b967b52/wde/workflows/core/executor/gpu_executor.py#L30C1-L46C30)


使用 chrome://tracing/ 查看 sync_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/sync1.png?raw=true" width="400">

> 图1 可以看到已经gpu利用率已经非常高了，没有什么空闲的空间

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/sync2.png?raw=true" width="400">

> 图2 放大对于7B-fp8的模型，一个step时间大概14ms，其中cpu部分比如调度预处理后处理才0.3ms

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/sync3.png?raw=true" width="400">

> 图3 1.5B的模型，一个step时间大概6ms，0.3ms也是不值一提

所以简单的实现异步调度几乎不会提高吞吐

### 简单异步调度

[简单调度代码](https://github.com/noooop/wde/blob/fb361f890bc2128175304d9788312be75b967b52/wde/workflows/core/executor/gpu_executor.py#L48C1-L60C32)


使用 chrome://tracing/ 查看 simple_async_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/simple_async1.png?raw=true" width="400">

> 图4 好像比图1密集了一点点

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/simple_async2.png?raw=true" width="400">

> 图5 确实gpu在做预处理时，gpu在执行，重叠的比例真的很小

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/simple_async3.png?raw=true" width="400">

> 图6 延迟口径是inference_time,  反应到吞吐延迟曲线上，大 max_num_batched_tokens 吞吐甚至会降低

### non_blocking

参考pytorch官方的 [non_blocking 教程](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)， 使用多个 cuda.Stream 结合 non_blocking 可以加速系统运行

直接跳到结论，通过多个cuda.Stream并行计算，不仅io和计算并行可以提高性能，两个batch计算并行也可以提高性能 

[non_blocking 异步调度代码](https://github.com/noooop/wde/blob/fb361f890bc2128175304d9788312be75b967b52/wde/workflows/core/executor/gpu_executor.py#L62C1-L93C26)

使用 chrome://tracing/ 查看 async_execute_loop.json

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/async1.png?raw=true" width="400">

> 图7 GPU空闲被两条重叠的Stream完全填补


<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/async2.png?raw=true" width="400">

> 图8 放大，不仅 io 和 计算 可以并行

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/async3.png?raw=true" width="400">

> 图9 小计算，甚至大计算之间也可以并行


<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/async4.png?raw=true" width="400">

> 图10 延迟口径是inference_time, 吞吐 async 确实比 sync 高一些

### double_buffer

[double_buffer 异步调度代码](https://github.com/noooop/wde/blob/fb361f890bc2128175304d9788312be75b967b52/wde/workflows/core/executor/gpu_executor.py#L95C1-L185C26)


<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/double_buffer.png?raw=true" width="400">

> 图11 延迟口径是inference_time, double_buffer 不会有任何提高，这里提一嘴主要是代码写得比较酷


### 代价是什么呢

异步调度本质上是交错调度两个批次的请求，所以对于每个请求，生成延迟会翻翻

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/latency-7b-fp8.png?raw=true" width="400">

> 图11 latency = finish_ts - scheduled_ts：从调度到完成的时间

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/latency-1.5b-bf16.png?raw=true" width="400">

> 图11 latency = finish_ts - scheduled_ts：从调度到完成的时间， 对于1.5b的小模型，加速效果要更好一些


### 与其他系统对比

跟 vllm-0.6.3.post1 和 sgl-0.3.5.post2 友善的对比一下

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/chat/profiling_decoding/throughput.png?raw=true" width="400">

> 图11 latency = finish_ts - scheduled_ts：从调度到完成的时间
> - vllm 和 sync 一模一样，毕竟wde的代码就是建立在vllm基础上的
> - sgl 确实要快一些，但不知道为什么 chunked_prefill_size 小于256会报错
> - 异步调度能提升吞吐，可以用更小的chunked_prefill_size达到性能饱和

## 未完待续




