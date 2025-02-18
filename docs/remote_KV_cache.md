# remote KV cache

## 引言

接着 [offloading KV cache](https://github.com/noooop/wde/blob/main/docs/offloading_KV_cache.md), 既然可以将内存作为 Prefix Caching 的大池子。

没有什么可以阻止把这个池子做的更大，做成分布式的。

实现了 remote KV cache 可以理解为 Prefix Caching 是 gpu cache、cpu cache、 remote cache 三级缓存。


## 基本测试

### 测试 memcpy

内存拷贝速度是block传输速度的基线，首先我们测试 block传输速度

- input_len = 8192
- max_num_batched_tokens = 1024
- block_size = 16

也就是 拷贝 8192 个token， 512 block 的速度

```commandline
python -m benchmarks.remote_kv_cache.baseline.test_memory_io
```

| model                | l2l naive (s) | b2b naive (s) | b2b fancy index (s) | b2b index_copy_ (s) | b2b cython memcpy (s) | swap in (s) | swap out (s) |
|----------------------|---------------|---------------|---------------------|---------------------|-----------------------|-------------|--------------|
| Qwen2.5-32B-Instruct | 0.277         | 0.130         | 0.335               | 0.336               | 0.165                 | 0.19        | 0.21         |
| Qwen2.5-7B-Instruct  | 0.088         | 0.033         | 0.077               | 0.077               | 0.059                 | 0.06        | 0.06         |
| Qwen2.5-3B-Instruct  | 0.090         | 0.024         | 0.049               | 0.049               | 0.031                 | 0.08        | 0.08         |
| glm-4-9b-chat-1m     | 0.126         | 0.045         | 0.107               | 0.108               | 0.056                 | 0.09        | 0.09         |
| Llama-3.1-8B         | 0.140         | 0.065         | 0.160               | 0.161               | 0.082                 | 0.09        | 0.10         |


> 表1 naive 的 copy 方法居然是最快的，甚至比 cython memcpy 都快。 可比口径下比 swap in 和 swap out 快一点

```python
to_kv_cache[t, ...] = from_kv_cache[f, ...]
```


### 测试 zmq 协议速度

```commandline
python -m benchmarks.remote_kv_cache.baseline.test_zmq_transfer
```

|                      | b2b naive (s) | tcp  | ipc  | inproc | delta-tcp | delta-ipc | delta-inproc |
|----------------------|---------------|------|------|--------|-----------|-----------|--------------|
| Qwen2.5-32B-Instruct | 0.13          | 0.56 | 0.57 | 0.13   | 4.27      | 4.37      | 1.02         |
| Qwen2.5-7B-Instruct  | 0.03          | 0.12 | 0.12 | 0.04   | 3.55      | 3.65      | 1.14         |
| Qwen2.5-3B-Instruct  | 0.02          | 0.10 | 0.07 | 0.03   | 3.95      | 2.93      | 1.11         |
| glm-4-9b-chat-1m     | 0.04          | 0.18 | 0.17 | 0.05   | 4.00      | 3.78      | 1.20         |
| Llama-3.1-8B         | 0.06          | 0.30 | 0.29 | 0.08   | 4.55      | 4.53      | 1.19         |


虽然 zeromq 号称 zero copy，但是 python 客户端还是需要多次copy，tcp和ipc速度差不多，大概等于四次拷贝，inproc差不多1次拷贝。

现在 python 客户端 至少要两次copy，已经给pyzmq提issues [Socket.recv_into](https://github.com/zeromq/pyzmq/issues/2057), 减少拷贝次数，欢迎围观。

这个项目朝着 分布式 出发，所以我们选择 tcp 协议。

### 测试 remote server 速度

进一步构造一个简单的能运行起来的zmq 客户端和服务器，端到端的测试速度

```commandline
python -m benchmarks.remote_kv_cache.baseline.test_server
```

|                      | b2b naive (s) | set deferred=False | set deferred=True | get  | stream_get | delta-set deferred=False | delta-set deferred=True | delta-get | delta-stream_get |
|----------------------|---------------|--------------------|-------------------|------|------------|--------------------------|-------------------------|-----------|------------------|
| Qwen2.5-32B-Instruct | 0.13          | 0.52               | 0.41              | 0.77 | 0.33       | 4.01                     | 3.16                    | 3.16      | 2.53             |
| Qwen2.5-7B-Instruct  | 0.03          | 0.13               | 0.09              | 0.24 | 0.07       | 3.76                     | 2.72                    | 2.72      | 2.10             |
| Qwen2.5-3B-Instruct  | 0.02          | 0.07               | 0.05              | 0.17 | 0.06       | 3.08                     | 2.15                    | 2.15      | 2.61             |
| glm-4-9b-chat-1m     | 0.04          | 0.15               | 0.10              | 0.29 | 0.09       | 3.33                     | 2.31                    | 2.31      | 2.06             |
| Llama-3.1-8B         | 0.06          | 0.27               | 0.21              | 0.50 | 0.13       | 4.17                     | 3.19                    | 3.19      | 1.96             |

进行了如下优化，将端到端速度优化到2~3次memcpy，在没有使用Socket.recv_into时，这是能做到的最快速度了
- set 使用 延迟写入，内存写入之前就返回成功
- get 使用 stream_get， 也就是返回多个block，流式一次返回一个，io和内存写入可以做Tiling

## 测试 transfer_out

swap_out 的 transfer 版， 验证 异步 transfer_out 几乎没有 overhead，不会拖慢系统。

```commandline
python -m benchmarks.remote_kv_cache.test_transfer_out
```

- input_len = 8192
- output_len = 1
- num_prompts = 4
- hit_rate = 0.
- max_num_batched_tokens_list = [1024, 768, 512, 384, 256, 128, 64, 32]


也就是 4 个 完全不同的 8192 个 token 的 prompt，是否使用 remote_kv_cache 也就是是否 transfer_out 有没有速度差异

开启 gpu_cache

| num_batched_tokens | swap_out-sync | swap_out-async-2 | transfer_out-sync | transfer_out-async-2 | Delta-sync | Delta-async-2 |
|--------------------|---------------|------------------|-------------------|----------------------|------------|---------------|
| 1024               | 2.45          | 2.59             | 2.37              | 2.46                 | 0.96       | 0.95          |
| 768                | 2.28          | 2.53             | 2.21              | 2.41                 | 0.97       | 0.95          |
| 512                | 2.31          | 2.50             | 2.24              | 2.51                 | 0.97       | 1.00          |
| 384                | 2.03          | 2.31             | 1.95              | 2.32                 | 0.96       | 1.00          |
| 256                | 1.74          | 2.20             | 1.69              | 2.14                 | 0.97       | 0.97          |
| 128                | 1.08          | 1.42             | 1.06              | 1.38                 | 0.98       | 0.97          |
| 64                 | 0.56          | 0.74             | 0.55              | 0.73                 | 0.99       | 1.00          |
| 32                 | 0.29          | 0.38             | 0.29              | 0.38                 | 0.99       | 0.99          |

关闭 gpu_cache，模拟 gpu_cache 被击穿

| num_batched_tokens | swap out-sync | swap out-async-2 | transfer_out-sync | transfer_out-async-2 | Delta-sync | Delta-async-2 |
|--------------------|---------------|------------------|-------------------|----------------------|------------|---------------|
| 1024               | 2.44          | 2.57             | 2.40              | 2.47                 | 0.98       | 0.96          |
| 768                | 2.29          | 2.51             | 2.23              | 2.43                 | 0.97       | 0.97          |
| 512                | 2.37          | 2.52             | 2.25              | 2.52                 | 0.95       | 1.00          |
| 384                | 2.07          | 2.30             | 1.95              | 2.34                 | 0.94       | 1.02          |
| 256                | 1.74          | 2.10             | 1.69              | 2.01                 | 0.97       | 0.96          |
| 128                | 1.08          | 1.41             | 1.07              | 1.39                 | 0.98       | 0.99          |
| 64                 | 0.56          | 0.74             | 0.56              | 0.73                 | 0.99       | 0.98          |
| 32                 | 0.29          | 0.38             | 0.28              | 0.38                 | 1.00       | 0.98          |

结论也跟 swap_out 类似
> - 无论开启 gpu_cache 还是关闭 gpu_cache, transfer_out 对比 swap_out overhead 都不大
> - 也就是几乎没有开销的可以把 kv cache 转移到另外一个进程（集群）
> - 使用 异步调度可以进一步降低 overhead，当然也可以进一步提高QPS


## 测试 test_transfer_in

这个测试为了进一步验证从另外一个进程（集群）拉 kv cache有没有用

- input_len = 1000
- output_len = 24
- num_prompts = 1000
- max_num_batched_tokens = 1024
- swap_space = 40
- hit_rate = [0.1 * x for x in range(0, 11)]

也就是 1000 个 输入 1000 token, 输出  24 个 token 的 prompt，随着前缀匹配的比例从0% 加到100%，QPS变化

运行两次，第一次将 kv cache 写入 remote，第二次可以从 remote 拉 kvcache

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.1/remote_kv_cache/test_transfer_in-2.png?raw=true" width="400">
> 图1 关闭 gpu_cache，模拟 gpu_cache 被击穿



<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.3.1/remote_kv_cache/test_transfer_in-1.png?raw=true" width="400">
> 图2 开启 gpu_cache


> 第二次可以从 remote 拉 kvcache，hit_rate = 0%的情况下， QPS 从 20 上升到 50~60， 开启 gpu_cache 可以让 qps进一步提高

## 真实场景测试

```commandline
python -m benchmarks.remote_kv_cache.test_long_prefill
```

- Qwen/Qwen2.5-7B-Instruct
- input_len = 1024 * 10
- num_prompts = 10
- max_num_batched_tokens = 1024
- output_len = N

模拟 1024 * 10 token 长度的 10 个不同 prompt 请求运行多轮

|                                | 2     | 4     | 8     | 16   | 32   | 64   | 128  |
|--------------------------------|-------|-------|-------|------|------|------|------|
| 1. remote+prefix_caching-1     | 0.88  | 0.88  | 0.86  | 0.81 | 0.72 | 0.59 | 0.43 |
| 2. remote+prefix_caching-2     | 47.94 | 24.92 | 12.64 | 6.35 | 3.18 | 1.59 | 0.80 |
| 3. remote+prefix_caching-3     | 48.69 | 24.91 | 12.64 | 6.36 | 3.18 | 1.59 | 0.80 |
| 4. remote+prefix_caching-4     | 5.54  | 5.03  | 4.13  | 4.01 | 2.63 | 1.44 | 0.75 |
| 5. remote+prefix_caching-5     | 48.28 | 24.78 | 12.60 | 6.35 | 3.18 | 1.59 | 0.80 |
| 6. remote+prefix_caching-6     | 48.80 | 24.98 | 12.63 | 6.35 | 3.18 | 1.59 | 0.80 |
| 7. remote+no_prefix_caching-1  | 0.89  | 0.89  | 0.86  | 0.81 | 0.72 | 0.59 | 0.43 |
| 8. remote+no_prefix_caching-2  | 9.26  | 7.71  | 5.80  | 4.68 | 2.91 | 1.52 | 0.78 |
| 9. remote+no_prefix_caching-3  | 9.26  | 7.71  | 5.80  | 4.69 | 2.91 | 1.52 | 0.78 |
| 10. remote+no_prefix_caching-4 | 5.56  | 5.00  | 4.19  | 4.07 | 2.65 | 1.44 | 0.76 |
| 11. remote+no_prefix_caching-5 | 9.27  | 7.72  | 5.78  | 4.69 | 2.92 | 1.52 | 0.78 |
| 12. remote+no_prefix_caching-6 | 9.28  | 7.67  | 5.78  | 4.69 | 2.92 | 1.52 | 0.78 |

> - 开启 gpu_cache 场景
> - remote+prefix_caching-1 硬算， 填充 remote_kv_cache
> - remote+prefix_caching-2、remote+prefix_caching-3 使用gpu cache
> - 重启 engine， 连上 remote_kv_cache
> - remote+prefix_caching-4，从 remote_kv_cache 拉取 
> - remote+prefix_caching-5、remote+prefix_caching-6 使用gpu cache

> - 关闭 gpu_cache，模拟 gpu_cache 被击穿
> - remote+no_prefix_caching-1-1 硬算， 填充 remote_kv_cache
> - remote+no_prefix_caching-2、remote+no_prefix_caching-3 使用 cpu cache
> - 重启 engine， 连上 remote_kv_cache
> - remote+no_prefix_caching-4，从 remote_kv_cache 拉取 
> - remote+no_prefix_caching-5、remote+no_prefix_caching-6 使用 cpu cache


gpu cache、cpu cache、 remote cache 三级缓存从快到慢。remote cache 也是相当可用的。



