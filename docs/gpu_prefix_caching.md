# Prefix Caching


## 引言
1. 显存管理
- LLM 输出总长度事先未知，kv cache 需要按需不断增加，为 llm 推理框架增加了不少麻烦。简单的显存管理会产生显存碎片（fragmentation）
- 受到操作系统的虚拟内存分页启发，vLLM 引入了 PagedAttention 减轻 kv cache 的显存碎片。
- 更多显存管理细节情参考
  - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
  - [vAttention](https://arxiv.org/abs/2405.04437)

2. sglang 使用 RadixTree 实现 Prefix Caching。参考[RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
3. vllm 使用 hash-based 实现 Prefix Caching。参考[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html)
4. Prefix Caching 只加速 prefill 阶段，对应提升首字延迟（First Token Latency、Time To First Token (TTFT)），不会影响decoding阶段，所以不会加速生成延迟（Decoding Latency、Time Per Output Token (TPOT)）。

## 实现 Prefix Caching

### [Naive](https://github.com/noooop/wde/tree/main/wde/workflows/decoding/kv_cache/naive)
不使用 Prefix Caching， overhead 最小, 只需要一个 physical_block_ids list 记录分配的 physical_block

### Prefix Caching by hash full blocks

与 vllm 使用 hash-based 实现 Prefix Caching 类似。
- 使用稳定的 hash 函数 md5 而不是 python 内置的 hash(), python 内置的 hash() 每次重启 python 相同的内容得到的输出不一样。
- 使用全局调度器，所有有相同前缀的请求都等待对应的 block 计算，每个 block 只算一次
- 只考虑满block。比如 block_size 为 16，每16个token为一个block，如果 prompt 长度不是 16的整数倍，比如最后一个block 只有15个 token，也就是不满的block 需要重新计算。



### YOCO(You only compute once) Prefix Caching
> 需要想个更好的名字

为了不重新计算不满的block，最大限度的避免重新计算，需要用前缀树（Trie）跟踪每个token。

这是个成本高收益极低的事情，比如 prompt 长度 900， block_size 为 16，有 56 个 full blocks，最后一个 block 有 4 个 token， 为了不重新计算这四个token，需要将所有token都用前缀树跟踪，成本非常高。



## 性能测试

```commandline
python -m benchmarks.prefix_caching.baseline
```

- input_len = 1000
- output_len = 24
- num_prompts = 1000
- max_num_batched_tokens = 1024
- swap_space = 40
- hit_rate = [0.1 * x for x in range(0, 11)]

也就是 1000 个 输入 1000 token, 输出  24 个 token 的 prompt，随着前缀匹配的比例从0% 加到100%，QPS变化


<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.1/prefix_caching.png?raw=true" width="400">

| hit rate | naïve-sync | naïve-async-2 | disable_prefix_caching-sync | disable_prefix_caching-async-2 | prefix_caching-sync | prefix_caching-async-2 | yoco-sync | yoco-async-2 |
|----------|------------|---------------|-----------------------------|--------------------------------|---------------------|------------------------|-----------|--------------|
| 0        | 21.03      | 21.74         | 20.70                       | 21.44                          | 20.62               | 20.93                  | 19.86     | 20.33        |
| 10       | 21.03      | 21.68         | 20.69                       | 22.18                          | 22.61               | 23.16                  | 21.80     | 22.70        |
| 20       | 21.01      | 21.60         | 20.73                       | 22.20                          | 25.05               | 25.78                  | 24.25     | 25.52        |
| 30       | 21.03      | 21.80         | 20.69                       | 21.45                          | 27.74               | 30.02                  | 26.90     | 28.50        |
| 40       | 21.02      | 21.83         | 20.72                       | 21.46                          | 31.09               | 33.43                  | 29.95     | 32.33        |
| 50       | 21.00      | 21.69         | 20.69                       | 21.40                          | 35.52               | 38.36                  | 34.15     | 37.16        |
| 60       | 21.01      | 21.83         | 20.69                       | 21.46                          | 39.87               | 43.76                  | 38.93     | 43.24        |
| 70       | 21.02      | 21.80         | 20.71                       | 22.19                          | 46.50               | 51.02                  | 45.84     | 51.67        |
| 80       | 21.00      | 21.78         | 20.73                       | 21.45                          | 58.16               | 65.92                  | 55.76     | 63.29        |
| 90       | 20.99      | 21.77         | 20.68                       | 21.35                          | 73.19               | 86.60                  | 71.77     | 85.70        |
| 100      | 20.98      | 21.83         | 20.69                       | 21.45                          | 98.84               | 124.47                 | 100.40    | 107.12       |


- 对比 所有 sync 和 async-2， 异步调度可以稍微增加 qps
- 对比 naïve-sync 和 disable_prefix_caching-sync (计算 block hash 但不进行caching)，计算 block hash 还是有一点点 overhead
- 对比 prefix_caching-sync 和 disable_prefix_caching-sync 在 hit rate=0% 场景，prefix caching overhead 是由 计算 block hash 引起的
- 对比 disable_prefix_caching-sync 和 prefix_caching-sync、disable_prefix_caching-async-2 和 prefix_caching-async-2 随着 前缀匹配的比例从0% 加到100%， QPS快速增加，异步调度增加的还要快一些
- 对比 prefix_caching-sync 和 yoco-sync 在 hit rate=0% 场景，使用前缀树跟踪每个 token，overhead 明显比计算 block hash 大。
- 对比 prefix_caching-sync 和 yoco-sync 在 hit rate=100% 场景，98.84 vs 100.40，yoco确实少算一些token，要快一点点
- 对比 prefix_caching-async-2 和 yoco-async-2，yoco 因为锁比较多，影响并行度，异步调度要慢一些

结论 hash full blocks 是工程上实现 Prefix Caching 比较好的选择


## 真实场景测试

```commandline
python -m benchmarks.prefix_caching.test_long_prefill
```

- Qwen/Qwen2.5-7B-Instruct
- input_len = 1024 * 10
- num_prompts = 10
- max_num_batched_tokens = 1024
- output_len = N

模拟 1024 * 10 token 长度的 10 个不同 prompt 请求运行多轮, 测试系统吞吐


| output_len                  | 2       | 4       | 8       | 16     | 32     | 64     | 128    | 
|-----------------------------|---------|---------|---------|--------|--------|--------|--------|
| test_naive                  | 0.9006  | 0.8956  | 0.8845  | 0.8349 | 0.7506 | 0.6152 | 0.4497 | 
| test_disable_prefix_caching | 0.8973  | 0.8941  | 0.8822  | 0.8297 | 0.7356 | 0.6049 | 0.441  | 
| test_prefix_caching-1       | 0.8962  | 0.8923  | 0.8796  | 0.8328 | 0.7407 | 0.6017 | 0.4403 | 
| test_prefix_caching-2       | 48.1283 | 25.0141 | 12.7597 | 6.4361 | 3.2322 | 1.6195 | 0.8095 | 
| test_prefix_caching-3       | 48.4216 | 25.0215 | 12.7616 | 6.4384 | 3.2305 | 1.6192 | 0.8095 | 
| test_yoco-1                 | 0.8857  | 0.8774  | 0.8661  | 0.8226 | 0.7288 | 0.5963 | 0.4355 | 
| test_yoco-2                 | 47.8238 | 25.0563 | 12.7427 | 6.4211 | 3.22   | 1.6077 | 0.8056 | 
| test_yoco-3                 | 48.4678 | 24.051  | 12.7356 | 6.3502 | 3.224  | 1.5974 | 0.8063 | 

命中 prefix_caching 可以显著提高系统吞吐，也就是加快响应速度
- 只加速 prefill 阶段，对应提升首字延迟（First Token Latency、Time To First Token (TTFT)）
- 不会影响decoding阶段，所以不会加速生成延迟（Decoding Latency、Time Per Output Token (TPOT)）

所以 prefix_caching 适合加速输入长输出短的场景，比如对同一个文档提问，或者非常多轮问答

## 未完待续





