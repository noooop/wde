
# 对 deepseek-r1 做推理优化

## 引言

由于 DeepSeek-V3 / R1 的专家数量众多，并且每层 256 个专家中仅激活其中 8 个。模型的高度稀疏性决定了必须采用很大的 overall batch size，才能给每个专家提供足够的 expert batch size，

deepseek-r1 模型架构
- embed_tokens -> 3 * MLP Layers -> 58 * MOE Layers -> lm_head 组成
- 每个 Layers 由 input_layernorm -> self_attn -> post_attention_layernorm -> mlp 组成
- self_attn 有 MHA 和 MLA 两种实现方式
- mlp 前三层是 dense mlp，后面 58 层是 moe

deepseek-v3/r1 单个 Expert 7168*2048*3 = 44M

58层moe 总共 58 * 256 * 44M ≈ 638 B

剩下的参数 657 B - 638 B ≈ 19B， 下称主干网络，主干网络包含共享专家。

所以最好的方法是主干网络和kvcache部署在一个集群，moe部署在另一个集群，主干网络通过rpc调用moe集群

> 如果使用现在主流的单机内做8卡TP机器间做pp，使用 H20 96Gx8 或者H800 80G x16， 没有多少空间给 kvcache
> 而且因为moe的稀疏性，没法在正常的batchsize（延迟）下将moe打到compute bound

而且dp+ep的方式每个节点都可以单独扩容缩容，负载平衡，故障转移

主干网络和moe集群可以分开单独优化，主干网络dp，moe集群ep

> Multi-head Latent Attention (MLA) 实际上就只有一个 head，要做 tp 需要退化到 Multi-head Attention

有两种优化思路：

1. 先尽量达到compute bound，提高吞吐，也就是优化大batchsize，然后减轻负载降低延迟

256 / 8  = 32, 所以最小需要32-way dp 才能将 ep 打到 compute bound。 

2. 先尽可能提高ep并行度，将延迟打下来，也就是优化batchsize=1，再考虑提高并行度

当大于 > 257-way EP 时，rpc延迟基本等于计算一个Expert的时间。而计算一个Expert的时间最快时间等于显存读取44M参数最快时间

单机内还可以对主干网络做8卡tp降低延迟，

除了compute bound和memory bound，还要考虑夸卡和夸机的通信带宽，大规模的ep有可能网卡带宽成为瓶颈。


最后rpc会引入比较大的通信开销，所以使用异步调度（双 batch 重叠）来掩盖通信开销，提高整体吞吐。

# 下面尝试在家搭（模拟）一个4090 32-way dp 的集群

## 每层基础性能测试 和 tuning block_wise kernel

下面对每个模块进行测试和调优

### embed_tokens

```commandline
python -m benchmarks.deepseek_v3.baseline.embed_tokens
```

| tokens | test_cpu (ms) | test_cpu_h2d (ms) | test_gpu (ms) | 
|--------|---------------|-------------------|---------------|
| 1      | 0.0034        | 0.0119            | 0.0094        |
| 2      | 0.004         | 0.0131            | 0.01          |
| 4      | 0.0052        | 0.0151            | 0.0113        |
| 8      | 0.0072        | 0.0253            | 0.0158        |
| 16     | 0.0113        | 0.0403            | 0.0189        |
| 32     | 0.0166        | 0.0776            | 0.0229        |
| 64     | 0.0293        | 0.1483            | 0.0332        |
| 128    | 0.0549        | 0.2221            | 0.0582        |
| 256    | 0.1086        | 0.3649            | 0.1166        |
| 512    | 0.506         | 0.7224            | 0.276         |
| 1024   | 1.1085        | 1.7059            | 0.7723        |
| 2048   | 2.2642        | 4.0767            | 2.2582        |


embed_tokens 占用 129280 * 7168 * bfloat16 = 1.7266 G 空间

将 embed_tokens 放在 内存节省 1.7 G显存，增加一倍的延迟，比如 1024 个 tokens，1.7059 ms vs 0.7723 ms

### mlp

```commandline
python -m benchmarks.deepseek_v3.baseline.mlp
```

