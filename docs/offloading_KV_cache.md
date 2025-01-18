# offloading KV cache

## 引言

## 基本测试

- prefill 是计算密集型操作，速度取决于GPU浮点运算速度
- kv cache 从 CPU 到 GPU 的交换，无论是cpu->GPU的 swap in 还是 GPU -> CPU 的 swap out，都是IO密集型操作，受限于pcie总线带宽
- GPU IO 和计算可以通过使用多个 cuda.Stream 结合 non_blocking 并行

如果 prefill 速度小于IO速度，GPU 计算开销可以将 kv cache 交换开销完全 cover.

```commandline
python -m benchmarks.offloading_KV_cache.test_prefills
python -m benchmarks.offloading_KV_cache.test_swap
```

首先测试 prefills 的速度， swap in 速度， swap out速度

- input_len = 8192
- max_num_batched_tokens = 1024

| model                          | num_attention_layers | num_kv_heads | head_size | kv_cache_dtype | prefill (s) | swap in (s) | swap out (s) |
|--------------------------------|----------------------|--------------|-----------|----------------|-------------|-------------|--------------|
| Qwen2.5-32B-Instruct-GPTQ-Int4 | 64                   | 8            | 128       | float16        | 3.69        | 0.17        | 0.18         |
| Qwen2.5-7B-Instruct-bf16       | 28                   | 4            | 128       | bfloat16       | 0.84        | 0.07        | 0.07         |
| Qwen2.5-7B-Instruct-fp8        | 28                   | 4            | 128       | bfloat16       | 0.53        | 0.07        | 0.07         |
| Qwen2.5-3B-Instruct-bf16       | 36                   | 2            | 128       | bfloat16       | 0.40        | 0.10        | 0.10         |
| glm-4-9b-chat-1m-bf16          | 40                   | 4            | 128       | bfloat16       | 1.08        | 0.11        | 0.11         |
| glm-4-9b-chat-1m-fp8           | 40                   | 4            | 128       | bfloat16       | 0.68        | 0.11        | 0.11         |
| Llama-3.1-8B-bf16              | 32                   | 8            | 128       | bfloat16       | 0.89        | 0.08        | 0.09         |
| Llama-3.1-8B-fp8               | 32                   | 8            | 128       | bfloat16       | 0.56        | 0.08        | 0.09         |


> 表1 可以看到常用的模型，甚至最小的 Qwen2.5-3B，一个 prefill GPU 计算时间都显著小于 kv cache 交换时间，这使得 offloading KV cache 成为可能.

让我们进一步看看 swap in 和 swap out 具体是怎么执行的。

## torch profile

```commandline
python -m benchmarks.offloading_KV_cache.profiler.swap_out 
```

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.0/offloading_KV_cache/swap_out.png?raw=true" width="400">

> 图1 Qwen2.5-3B-Instruct-bf16 swap out 1024 个 token 花费 16ms， 8192 个token 花费大概 128 ms，跟表1 里的 0.10s 差不多


```commandline
python -m benchmarks.offloading_KV_cache.profiler.swap_in
```

- input_len = 1024 * 2
- output_len = 16
- num_prompts = 4
- max_num_batched_tokens = 1024
- 关闭 gpu prefix_caching

运行4次相同的 2048 个 token 的 prompt, 其中第一次需要计算并swap out， 后面三次命中offloading_KV_cache， swap in

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.0/offloading_KV_cache/swap_in.png?raw=true" width="400">


> 图2 计算和swap使用两个stream, swap 几乎不干扰计算， 看上去挺不错的


## overhead

```commandline
python -m benchmarks.offloading_KV_cache.profiler.test_long_prefill
```

- Qwen/Qwen2.5-7B-Instruct
- input_len = 1024 * 10
- num_prompts = 10
- max_num_batched_tokens = 1024
- output_len = N

将 1024 * 10 token 长度的 10 个不同 prompt 运行多轮，其中第一轮需要计算，第二轮开始命中cache


| N   | naive (qps) | prefix_caching-1 (qps) | prefix_caching-2 (qps) | offloading+prefix_caching-1 (qps) | offloading+prefix_caching-2 (qps) | offloading+no_prefix_caching-1 (qps) | offloading+no_prefix_caching-2 (qps) | overhead (s) |
|-----|-------------|------------------------|------------------------|-----------------------------------|-----------------------------------|--------------------------------------|--------------------------------------|--------------|
| 2   | 0.89        | 0.90                   | 49.65                  | 0.90                              | 49.07                             | 0.91                                 | 8.50                                 | 0.97         |
| 4   | 0.90        | 0.90                   | 25.38                  | 0.90                              | 25.14                             | 0.90                                 | 7.23                                 | 0.99         |
| 8   | 0.89        | 0.89                   | 12.82                  | 0.89                              | 12.66                             | 0.89                                 | 5.71                                 | 0.96         |
| 16  | 0.85        | 0.84                   | 6.43                   | 0.84                              | 6.37                              | 0.84                                 | 3.95                                 | 0.96         |
| 32  | 0.75        | 0.75                   | 3.22                   | 0.74                              | 3.19                              | 0.74                                 | 2.45                                 | 0.95         |
| 64  | 0.62        | 0.61                   | 1.61                   | 0.60                              | 1.60                              | 0.60                                 | 1.39                                 | 0.95         |
| 128 | 0.45        | 0.44                   | 0.81                   | 0.44                              | 0.81                              | 0.44                                 | 0.74                                 | 1.07         |

> 表2
> - 对比 naive、prefix_caching-1、offloading+prefix_caching-1、offloading+no_prefix_caching-1：prefix_caching 和 offloading 几乎没有 overhead
> - 对比 prefix_caching-2 和 offloading+prefix_caching-2， 如果内命中gpu的prefix_caching，offloading 几乎没有 overhead
> - 对比 offloading+prefix_caching-2 和 offloading+no_prefix_caching-2， 关闭 gpu 的 prefix_caching，也就是命中cache 需要 swap in，qps明显变低
> - 可以计算出 offloading+prefix_caching-2 和 offloading+no_prefix_caching-2 对比的 overhead， 随着输出序列长度的增加几乎不变

overhead 是从哪里来的?

- 当第一轮没有命中cache时，计算同时 swap out
- input_len = 1024 * 10， max_num_batched_tokens = 1024， 一个请求需要10个 prefill step
- 执行模型之前kv cache还没有填充， swap out 是从执行模型结束才开始，采样和后处理比swap out慢，所以 swap out 结束，并可以让其他请求命中 swap in是下一个 step 结束
- swap out 可以一边 计算，一边 swap out，几乎不影响
- swap in 需要将所有命中 cache 的 block 都 swap in 才能开始执行
- 所以 swap in 有一些 overhead 

- 总之 使用 cpu kv cache 速度比 gpu kv cache慢，有等待block swap，以及将所有需要的 block swap in 的 overhead

## 实现细节

暂时使用 hash full blocks 实现 cpu kv cache，也就是只管写满了的block。

hash full blocks 是工程上实现 Prefix Caching 比较好的选择。

## 未完待续