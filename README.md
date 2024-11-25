# 介绍
- Workflow Defined Engine, 针对不同架构的模型实现不同的模块，并按需加载所需的模块。
- Asynchronous scheduling, 异步调度，提高GPU利用率，增加吞吐降低延迟
- Microservices linked by Zeromq，减少序列化反序列化和io开销，进一步增加吞吐降低延迟

# 文档
- [配环境&安装](./setup)
- [支持的模型](./docs/supported_models.md)
- [对 prefill only models 推理性能调优](./docs/performance_tuning_for_prefill_only_models.md)
- [对 Decoding (chat) models 推理性能调优](./docs/performance_tuning_for_decoding_models.md)

# 应用
- [快速入门](./docs/quickstart.md)
- [chat_client](./applications/chat_cli)
- [multi-agent](./applications/agents)
- [与 ollama 和 openai 兼容的 webserver](https://github.com/noooop/wde/tree/main/examples/webserver)

# Acknowledgement
- [vLLM](https://github.com/vllm-project/vllm)


