# offloading KV cache

## 引言

GPU 显存速度快，但是又贵又小，CPU 内存速度慢，但便宜量又大

如何能让 cpu kv cache 平替宝贵的 GPU kv cache 呢，这就是 offloading KV cache。


## 基本测试

- prefill 是计算密集型操作，速度取决于GPU浮点运算速度
- kv cache 从 CPU 到 GPU 的交换，无论是cpu->GPU的 swap in (H2D) 还是 GPU -> CPU 的 swap out (D2H)，都是IO密集型操作，受限于pcie总线带宽
- GPU IO 和计算可以通过使用多个 cuda.Stream 结合 non_blocking 并行

如果 prefill 速度小于IO速度，GPU 计算开销可以将 kv cache swap 开销完全 cover. 实现 Computation-Communication Overlap.

```commandline
python -m benchmarks.offloading_KV_cache.baseline.test_prefills
python -m benchmarks.offloading_KV_cache.baseline.test_swap
```

首先测试 prefills 的速度， swap in 速度， swap out速度

- input_len = 8192
- max_num_batched_tokens = 1024

| model                          | num_attention_layers | num_kv_heads | head_size | kv_cache_dtype | prefill (s) | swap in (s) | swap out (s) |
|--------------------------------|----------------------|--------------|-----------|----------------|-------------|-------------|--------------|
| Qwen2.5-32B-Instruct-GPTQ-Int4 | 64                   | 8            | 128       | float16        | 3.69        | 0.19        | 0.21         |
| Qwen2.5-7B-Instruct-bf16       | 28                   | 4            | 128       | bfloat16       | 0.84        | 0.06        | 0.06         |
| Qwen2.5-7B-Instruct-fp8        | 28                   | 4            | 128       | bfloat16       | 0.53        | 0.06        | 0.06         |
| Qwen2.5-3B-Instruct-bf16       | 36                   | 2            | 128       | bfloat16       | 0.40        | 0.08        | 0.08         |
| glm-4-9b-chat-1m-bf16          | 40                   | 4            | 128       | bfloat16       | 1.08        | 0.09        | 0.09         |
| glm-4-9b-chat-1m-fp8           | 40                   | 4            | 128       | bfloat16       | 0.68        | 0.09        | 0.09         |
| Llama-3.1-8B-bf16              | 32                   | 8            | 128       | bfloat16       | 0.89        | 0.09        | 0.10         |
| Llama-3.1-8B-fp8               | 32                   | 8            | 128       | bfloat16       | 0.56        | 0.09        | 0.10         |


> 表1 可以看到常用的模型，甚至最小的 Qwen2.5-3B，一个 prefill GPU 计算时间都显著小于 kv cache 交换时间，这使得 offloading KV cache 成为可能.

让我们进一步看看 swap in 和 swap out 具体是怎么执行的。

## torch profile

### profiler swap_out

```commandline
python -m benchmarks.offloading_KV_cache.profiler.swap_out 
```

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.0/offloading_KV_cache/swap_out.png?raw=true" width="400">

> 图1 Qwen2.5-3B-Instruct-bf16 swap out 1024 个 token 花费 16ms， 8192 个token 花费大概 128 ms，跟表1 里的 0.08s 对比, 估计 swap 速度还是有一些改进空间.

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.1/offloading_KV_cache/swap_out-2.png?raw=true" width="400">

> 图2 放大，swap out 对每层，每个kv block都 launch 一个 Memcpy DtoH cuda kernel，比较低效。但是暂时没有什么更好的办法。

> kv cache 布局
> - 为了支持 Pipeline-Parallelism， gpu kvcache 使用 layerwise 布局，也就是 num_attention_layers * (2, num_blocks, block_size, num_kv_heads, head_size)
> - 为了方便 block 交换，CPU kvcache 使用 blockwise 布局， 也就是 (num_blocks, num_attention_layers, 2, block_size, num_kv_heads, head_size)

### profiler swap_in

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


> 图3 计算和swap使用两个stream, swap 几乎不干扰计算， 看上去挺不错的


## 测试 swap_out
这个测试为了进一步验证 异步 swap_out 几乎没有 overhead，不会拖慢系统。

```commandline
python -m benchmarks.offloading_KV_cache.test_swap_out
```

- input_len = 8192
- output_len = 1
- num_prompts = 4
- hit_rate = 0.
- max_num_batched_tokens_list = [1024, 768, 512, 384, 256, 128, 64, 32]


也就是 4 个 完全不同的 8192 个 token 的 prompt，是否使用 offloading 也就是是否 swap_out 有没有速度差异

开启 gpu_cache

| num_batched_tokens | sync | async-2 | offloading-sync | offloading-async-2 | Delta-sync | Delta-async-2 |
|--------------------|------|---------|-----------------|--------------------|------------|---------------|
| 1024               | 2.48 | 2.58    | 2.45            | 2.59               | 0.99       | 1.00          |
| 768                | 2.32 | 2.52    | 2.28            | 2.53               | 0.99       | 1.00          |
| 512                | 2.40 | 2.57    | 2.31            | 2.50               | 0.96       | 0.97          |
| 384                | 2.12 | 2.36    | 2.03            | 2.31               | 0.96       | 0.98          |
| 256                | 1.83 | 2.16    | 1.74            | 2.20               | 0.95       | 1.02          |
| 128                | 1.14 | 1.44    | 1.08            | 1.42               | 0.95       | 0.99          |
| 64                 | 0.58 | 0.75    | 0.56            | 0.74               | 0.96       | 0.98          |
| 32                 | 0.30 | 0.39    | 0.29            | 0.38               | 0.96       | 0.98          |

关闭 gpu_cache

| num_batched_tokens | sync | async-2 | offloading-sync | offloading-async-2 | Delta-sync | Delta-async-2 |
|--------------------|------|---------|-----------------|--------------------|------------|---------------|
| 1024               | 2.49 | 2.56    | 2.44            | 2.57               | 0.98       | 1.00          |
| 768                | 2.32 | 2.54    | 2.29            | 2.51               | 0.99       | 0.99          |
| 512                | 2.40 | 2.54    | 2.37            | 2.52               | 0.99       | 0.99          |
| 384                | 2.11 | 2.37    | 2.07            | 2.30               | 0.98       | 0.97          |
| 256                | 1.81 | 2.18    | 1.74            | 2.10               | 0.96       | 0.96          |
| 128                | 1.11 | 1.44    | 1.08            | 1.41               | 0.98       | 0.98          |
| 64                 | 0.58 | 0.75    | 0.56            | 0.74               | 0.96       | 0.99          |
| 32                 | 0.31 | 0.39    | 0.29            | 0.38               | 0.94       | 0.98          |

> 表2 无论开启 gpu_cache 还是关闭 gpu_cache, swap_out overhead 都不大
> - 也就是几乎没有开销的可以把 kv cache 转移到更大的 cpu 内存里
> - num_batched_tokens 越大 overhead 越小
> - 使用 异步调度可以进一步降低 overhead，当然也可以进一步提高QPS


## 测试 swap_in
这个测试为了进一步验证 cpu kv cache 异步 swap in，可以多大程度平替宝贵的 GPU kv cache

```commandline
python -m benchmarks.offloading_KV_cache.test_swap_in
```

- input_len = 1000
- output_len = 24
- num_prompts = 1000
- max_num_batched_tokens = 1024
- swap_space = 40
- hit_rate = [0.1 * x for x in range(0, 11)]

也就是 1000 个 输入 1000 token, 输出  24 个 token 的 prompt，随着前缀匹配的比例从0% 加到100%，QPS变化

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.1/offloading_KV_cache/test-swap_in.png?raw=true" width="400">

> 图4 
> - 前缀匹配的比例 <80%, Computation-Communication Overlap, cpu kv cache 和 cpu kv cache QPS几乎一样，也就是 cpu kv cache 平替 GPU kv cache
> - 超过 80%，也就是先从 cpu kv cache swap_in 80% 的内容，然后 prefill 20%内容， decoding 24 个 token，后一半计算时间没办法Overlap前一段swap_in的时间
> - 总之，offloading KV cache 是非常好用的

## 真实场景测试

```commandline
python -m benchmarks.offloading_KV_cache.profiler.test_long_prefill
```

- Qwen/Qwen2.5-7B-Instruct
- input_len = 1024 * 10
- num_prompts = 10
- max_num_batched_tokens = 1024
- output_len = N

这里模拟 1024 * 10 token 长度的 10 个不同 prompt 运行多轮，其中第一轮需要计算，第二轮开始命中cache

分别测试 naive（不使用prefix_caching），使用 prefix_caching， offloading+prefix_caching， offloading不使用prefix_caching 四种场景

| output_len:                     | 2     | 4     | 8     | 16   | 32   | 64   | 128  |
|---------------------------------|-------|-------|-------|------|------|------|------|
| 1. naive                        | 0.91  | 0.90  | 0.89  | 0.84 | 0.75 | 0.62 | 0.45 |
| 2. prefix_caching-1             | 0.90  | 0.90  | 0.88  | 0.84 | 0.74 | 0.61 | 0.44 |
| 3. prefix_caching-2             | 49.50 | 25.48 | 12.86 | 6.43 | 3.23 | 1.62 | 0.81 |
| 4. prefix_caching-3             | 49.96 | 25.46 | 12.86 | 6.39 | 3.23 | 1.62 | 0.81 |
| 5. offloading+prefix_caching-1  | 0.90  | 0.87  | 0.86  | 0.81 | 0.72 | 0.59 | 0.43 |
| 6. offloading+prefix_caching-2  | 48.86 | 25.04 | 12.67 | 6.36 | 3.19 | 1.60 | 0.80 |
| 7. offloading+prefix_caching-3  | 49.22 | 25.01 | 12.66 | 6.36 | 3.19 | 1.60 | 0.80 |
| 8. offloading+no_gpu_caching-1  | 0.90  | 0.88  | 0.86  | 0.82 | 0.72 | 0.59 | 0.43 |
| 9. offloading+no_gpu_caching-2  | 9.23  | 7.69  | 5.79  | 4.69 | 2.92 | 1.52 | 0.78 |
| 10. offloading+no_gpu_caching-3 | 9.23  | 7.69  | 5.79  | 4.69 | 2.90 | 1.52 | 0.78 |

> 表2
> - 第1行 naive 为不使用 gpu cache，硬算的速度，作为比较的基线
> - 第2~4行 为使用 prefix_caching，第2行需要计算，第3~4行 命中gpu_cache，速度有明显提升
> - 第5~7行 为使用 offloading+prefix_caching，第5行需要计算，第3~4行 命中gpu_cache，速度跟prefix_caching几乎相同
> - 第8~10行 offloading不使用prefix_caching，第8行需要计算, 第3~4行 命中cpu_cache，速度跟 gpu_cache 肯定是 没法比，但随着输出token的增加，Computation-Communication Overlap优化更明显，速度差异不是那么显著
> - 对比 naive、prefix_caching-1、offloading+prefix_caching-1、offloading+no_prefix_caching-1：prefix_caching 和 offloading 几乎没有 overhead
> - 对比 prefix_caching-2 和 offloading+prefix_caching-2， 如果内命中gpu的prefix_caching，offloading 几乎没有 overhead
> - 对比 offloading+prefix_caching-2 和 offloading+no_prefix_caching-2， 关闭 gpu 的 prefix_caching，也就是命中cache 需要 swap in，qps明显变低


cpu cache 和 cpu cache 具体速度差异从哪里来?

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