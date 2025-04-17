# 使用 remote_KV_cache



## [安装](https://github.com/noooop/wde/tree/main/setup)

## 相关文档

- [实现 prefix caching](https://github.com/noooop/wde/blob/main/docs/gpu_prefix_caching.md)
- [实现 offloading prefix caching](https://github.com/noooop/wde/blob/main/docs/offloading_KV_cache.md)
- [实现 remote KV cache](https://github.com/noooop/wde/blob/main/docs/remote_KV_cache.md)
- [实现 persistence kv cache](https://github.com/noooop/wde/blob/main/docs/persistence_kv_cache.md)


## 使用

1. 一键部署 "THUDM/glm-4-9b-chat-1m-hf"  和 相应的remote_kv_cache

```commandline
wde serving examples/use_remote_KV_cache/deploy.yml
```

THUDM/glm-4-9b-chat-1m-hf 长上下文能力不错，使用fp8部署占用 10.1515 GB 显存，如果使用 24G的 4090 给 KV cache 留 10.5519 GB。

换算一下大概可以支撑 1310720 的上下文

```
chat:
  models:
    -
      model: "THUDM/glm-4-9b-chat-1m-hf"  # 模型名称
      engine_args:
        gpu_memory_utilization: 0.9    # GPU显存利用率
        quantization: "fp8"            # fp8 推理
        block_size: 16                 # block_size 默认 16
        max_num_requests: 1            # 最大并行请求数
        max_num_batched_tokens: 1024   # Chunked Fill 最大 batched_tokens 数
        max_model_len: 110000          # 最大上下文长度
        swap_space: 20                 # cpu swap_space 内存大小 G
        enable_prefix_caching: True    # 打开 gpu prefix_caching
        remote_kv_cache_server: True   # 打开 remote_kv_cache_server
        trust_remote_code: True        # THUDM/glm-4-9b-chat-1m-hf 需要 trust_remote_code: True
        default_options:
          max_tokens: 1024             # 最大输出大小

remote_kv_cache:
  -
    model: "THUDM/glm-4-9b-chat-1m-hf"    # remote_kv_cache 模型名称， 需要和chat模型对应
    engine_args:
      block_size: 16                   # block_size 默认 16， 需要和chat模型对应
      memory_space: 20                 # kv cache 使用 cpu 内存大小 G
      file_space: 100                  # kv cache 使用 ssd 大小 G
      kv_cache_folder: "/share/test_kv_cache"    # kv cache 保存目录
 
entrypoints: ["ollama_compatible", "openai_compatible"]   # 启动 ollama 和 openai server
```

也就是使用 THUDM/glm-4-9b-chat-1m-hf 模型， 上下文长度 110000， remote_kv_cache 最大 占内存 20G， 占 file_space 100G


2. 生成一个长文档

```commandline
python -m examples.use_remote_KV_cache.make_dummy_inputs --model "THUDM/glm-4-9b-chat-1m-hf" --length 100000 --filename dummy.txt
```

生成大概长度10万 token 的长文档


3. 首字延迟（First Token Latency、Time To First Token (TTFT)）

```commandline
python -m examples.use_remote_KV_cache.speed_test --model "THUDM/glm-4-9b-chat-1m-hf" --filename dummy.txt
```

- 第一次运行，没有任何缓存  34.79032147899852 s
- 第二次运行，使用gpu缓存 0.3699891229989589 s amazing!
- 将 wde serving 重启，
  - 日志 recover 6514 blocks. info: {'block_size': 16, 'num_blocks': 81920, 'num_full_blocks': 6514, 'num_free_blocks': 75406}
  - 也就是从硬盘中恢复 6514 个 blocks, 6514 * 16 = 104224，差不多
- 第三次运行，使用 ssd 缓存 6.42922614300187 s amazing! 此时 ssd 上的文件大小为 7.95G

使用 7.95G 硬盘缓存 可以让 TTFT 从 34s 变为 6s。现在2T的 ssd 才 899，赶快划 1T 作为 kv cache。

再考虑到，从 remote_KV_cache 拉取时，gpu处于空闲状态，还可以做其他事情，使用remote_KV_cache吞吐也可以大幅提升。

更多信息 请参考 [缓存增强生成 Cache-Augmented Generation](https://arxiv.org/abs/2412.15605)

4. 压力测试

- gpu 显存 KV cache 留 10.5519 GB， 大概可以支撑 1310720 的上下文
- 执行器 的 cpu cache 20G, 大概缓存 16384 * 16 = 262144 个 token
- remote_KV_cache sever cpu cache 20G 也是 262144 个 token
- 也就是 2 个 10万 token 的长文档可以击穿 gpu_cache, 3个 10万 token 的长文档可以击穿 cpu cache，只能 ssd 硬抗，那就给ssd上点压力吧

```commandline
python -m examples.use_remote_KV_cache.stress_test --model "THUDM/glm-4-9b-chat-1m-hf" --length 100000 --filename dummy.txt --n 6
```

| n | 1     | 2     | 3     | 4     | 5     | 6     |
|---|-------|-------|-------|-------|-------|-------|
| 1 | 33.94 |       |       |       |       |       |
| 2 | 0.33  | 33.97 |       |       |       |       |
| 3 | 1.25  | 1.18  | 33.89 |       |       |       |
| 4 | 3.49  | 2.48  | 2.59  | 33.88 |       |       |
| 5 | 5.76  | 5.84  | 6.20  | 6.24  | 33.81 |       |
| 6 | 5.91  | 6.33  | 6.29  | 5.90  | 5.94  | 34.36 |


- 硬算 33 s
- 命中 gpu cache 0.33 s
- 命中 执行器 cpu cache ~1 s
- 命中 remote_KV_cache cpu cache 2-3 s
- 命中 remote_KV_cache ssd cache 5-6 s
- 执行完硬盘占用大概 47.6G

请大家弹幕告诉我这 47.6G 值不值