# Prefix Caching


## 引言
1. 显存管理
- LLM 输出总长度事先未知，kv cache 需要按需不断增加，为 llm 推理框架增加了不少麻烦。简单的显存管理会产生显存碎片（fragmentation）
- 受到操作系统的虚拟内存分页启发，vLLM 引入了 PagedAttention 减轻 kv cache 的显存碎片。
- 更多显存管理细节情参考
  - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
  - [vAttention](https://arxiv.org/abs/2405.04437)

2. [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
> RadixAttention, enables the automatic reuse of the KV cache across
multiple generation calls. In existing inference engines, the KV cache of a request is discarded
after processing is completed, preventing the KV cache from being reused across multiple calls and
significantly slowing down the execution. Instead, our system maintains an LRU cache of the KV
cache for all requests within a radix tree. This approach manages the KV cache as a traditional cache
and uses a radix tree for efficient matching, insertion, and eviction. It allows the runtime to handle
various reuse patterns with a cache-aware scheduling policy efficiently. The second technique is a
compressed finite state machine, which enables faster constrained decoding for structured outputs.
Existing systems follow the constraints only for the next token by masking probabilities of disallowed
tokens, making them able to decode only one token at a time. Instead, our system analyzes the
constraints and builds a compressed finite-state machine to represent the constraint. This approach
compresses a multi-token path into a single-step path whenever possible, allowing the decoding of
multiple tokens at once to achieve faster decoding speed.

## 实现 Prefix Caching

### Naive

### Prefix Caching by hash full blocks

### YOCO(You only compute once) Prefix Caching


## 性能测试

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/wde/0.2.3/prefix_caching.png?raw=true" width="400">

同步 sync
- 随着 公共前缀长度从0-100%
- 吞吐(横轴) 快速增加
- 延迟(纵轴) 90%前变化不大，100%极速下降
- YOCO 开销大，所以延迟(纵轴) 比 hash full blocks 稍微高一点
- YOCO 少计算一些token，所以吞吐(横轴) 比 hash full blocks 高一些

异步 async-2
- hash full blocks 吞吐高，延迟低
- 虽然 YOCO 少计算一些token，但没有满的block占比很少，所以提高不大
- YOCO 锁比 hash full blocks 少一些，所以并行度高一些

结论 hash full blocks 是工程上实现 Prefix Caching 比较好的选择



## 未完待续





