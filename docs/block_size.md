# block size

## 引言

PagedAttention 不同 block size 对推理性能是否有影响

## vllm_flash_attn

> Paged KV cache block size must be divisible by 16

所以 vllm_flash_attn 最小的 block size 为 16

```commandline
python -m benchmarks.chat.test_block_size
```


| block_size                   | prefill-16 | prefill-32 | prefill-64 | prefill-128 | decoding-16 | decoding-32 | decoding-64 | decoding-128 |
|------------------------------|------------|------------|------------|-------------|-------------|-------------|-------------|--------------|
| Qwen/Qwen2.5-7B-Instruct     | 0.9112     | 0.906      | 0.9098     | 0.9078      | 0.3054      | 0.3055      | 0.305       | 0.3053       |
| Qwen/Qwen2.5-3B-Instruct     | 1.8555     | 1.8509     | 1.8655     | 1.8675      | 0.5974      | 0.5994      | 0.5997      | 0.5988       |
| Qwen/Qwen2.5-7B-Instruct-fp8 | 1.387      | 1.3904     | 1.3894     | 1.391       | 0.4574      | 0.4578      | 0.4577      | 0.457        |


调整 block size 对 prefill 和 decoding 几乎没有影响

## [flashinfer-v02](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)

> FlashInfer’s standout feature is its highly flexible block-sparse FlashAttention implementation, supporting any block size configuration.
> We compared two attention implementations: PageAttention with page_size=1 (use vector-sparse attention implementation) and variable-length dense attention.

<img src="https://flashinfer.ai/assets/imgs/fa3-template.png" width="400">


sglang 使用 page_size=1 的 flashinfer, 可以实现细粒度的 kv cache 管理

< 这里应该有一个使用 flashinfer-v02 的对比测试 >


## 未完待续

