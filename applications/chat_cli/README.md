# [配置环境](https://github.com/noooop/wde/tree/main/setup)

## 帮助
```
$ python -m applications.chat_cli
Usage: python -m applications.chat_cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  run
```

## run 运行模型
```
$ python -m applications.chat_cli run Qwen/Qwen2.5-7B-Instruct
INFO 11-24 15:31:25 server.py:94 ZeroNameServer: InMemoryNameServer running! port: 55754.
INFO 11-24 15:31:25 server.py:25 ZeroManager for RootZeroManager running! port: 59748
正在加载模型...
INFO 11-24 15:31:27 config.py:34 Chunked prefill is enabled with max_num_batched_tokens=512.
INFO 11-24 15:31:27 config.py:80 Initializing an LLM engine (v0.2.0) with config: model='Qwen/Qwen2.5-7B-Instruct', tokenizer='Qwen/Qwen2.5-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, quantization=None, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, seed=0, served_model_name=Qwen/Qwen2.5-7B-Instruct, enable_prefix_caching=False, scheduling=async)
INFO 11-24 15:31:27 llm_engine.py:56 Use async scheduling
INFO 11-24 15:31:27 selector.py:54 Using FLASH ATTN backend.
INFO 11-24 15:31:27 model_runner.py:67 Starting to load model Qwen/Qwen2.5-7B-Instruct...
INFO 11-24 15:31:28 weight_utils.py:224 Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:00,  3.49it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:00<00:00,  3.21it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:00<00:00,  3.11it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:01<00:00,  3.05it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:01<00:00,  3.11it/s]

INFO 11-24 15:31:30 gpu_worker.py:98 init gpu free memory 23.0994 GB
INFO 11-24 15:31:30 gpu_worker.py:106 Loading model weights took 14.3633 GB
INFO 11-24 15:31:30 gpu_worker.py:129 After profile run: Peak Memory 15.5488 GB, of which Runtime Memory 1.1855 GB, 5.7004 GB leave for KV cache
INFO 11-24 15:31:30 gpu_executor.py:78 # GPU blocks: 6671, # CPU blocks: 4681
INFO 11-24 15:31:32 zero_engine.py:43 ZeroEngine Qwen/Qwen2.5-7B-Instruct is running! port: 56311
加载完成!
!quit 退出, !next 开启新一轮对话。玩的开心！       
================================================================================                                                                                                                                         ================================================================================
[对话第1轮]
(用户输入:)
hello
(Qwen/Qwen2.5-7B-Instruct:)

Hello! How can I assist you today?

[对话第2轮]
(用户输入:)
你会说中文吗？
(Qwen/Qwen2.5-7B-Instruct:)

会呀！我能说中文。有什么我可以帮助你的吗？

[对话第3轮]
(用户输入:)
用中文讲个笑话吧
(Qwen/Qwen2.5-7B-Instruct:)

当然可以！这里有一个简单的笑话：

为什么电脑经常生病？

因为它窗户（Windows）太多！

[对话第4轮]
(用户输入:)
!quit
INFO 11-24 15:32:32 server.py:106 ZeroNameServer is clean_up!
INFO 11-24 15:32:34 server.py:106 ZeroEngine is clean_up!
INFO 11-24 15:32:34 server.py:106 ZeroManager is clean_up!
quit gracefully
```
