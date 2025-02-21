# persistence kv cache

## 引言

接着 [remote KV cache](https://github.com/noooop/wde/blob/main/docs/remote_KV_cache.md), 既然使用在另一个进程（集群）的内存作为 Prefix Caching 的大池子。

何不将硬盘(ssd)也用起来，又大又便宜，重启服务也不会丢数据。

也就是 remote KV cache 支持内存 + ssd 混合模式。


## 基本测试

测试 ssd 读写速度，

> 硬件：宏碁掠夺者（PREDATOR）2TB SSD固态硬盘 M.2,  PCIe4.0，读速7400MB/s

- input_len = 8192
- max_num_batched_tokens = 1024
- block_size = 16

也就是 拷贝 8192 个token， 512 block 的速度

```commandline

python -m benchmarks.persistence_kv_cache.baseline.test_mmap
python -m benchmarks.persistence_kv_cache.baseline.test_filesystem
python -m benchmarks.persistence_kv_cache.baseline.test_leveldb
```

| model                | b2b naive (s) | mmap write | mmap read | filesystem write | filesystem read | leveldb write | leveldb read | Delta-filesystem write | Delta-filesystem read | 
|----------------------|---------------|------------|-----------|------------------|-----------------|---------------|--------------|------------------------|-----------------------|
| Qwen2.5-32B-Instruct | 0.130         | 0.623      | 0.636     | 0.556            | 0.473           | 2.700         | 0.492        | 4.274                  | 3.641                 | 
| Qwen2.5-7B-Instruct  | 0.033         | 0.139      | 0.154     | 0.126            | 0.128           | 0.508         | 0.149        | 3.806                  | 3.872                 |
| Qwen2.5-3B-Instruct  | 0.024         | 0.091      | 0.093     | 0.085            | 0.087           | 0.301         | 0.082        | 3.532                  | 3.613                 | 
| glm-4-9b-chat-1m     | 0.045         | 0.193      | 0.218     | 0.180            | 0.176           | 0.718         | 0.164        | 3.993                  | 3.907                 | 
| Llama-3.1-8B         | 0.065         | 0.308      | 0.314     | 0.270            | 0.262           | 1.110         | 0.260        | 4.158                  | 4.035                 | 


> 表1
> - filesystem 读写都很快
> - mmap 读写比 filesystem 略慢
> - leveldb 写非常慢， 估计是写 WAL 花一些时间，读跟 filesystem 接近
> - filesystem 读写对比内存读写，慢3-4倍，已经非常可用了，如果使用 PCIe5.0 的 ssd 速度还可以提升一倍

通过 filesystem 和 mmap 对比，这个速度就是硬盘最快的随机读写速度，之后会测试其他方案的速度，寻找更好的 persistence kv cache 实现方式。
- 测试不同文件系统
- kvdb: leveldb, RocksDB
- sql: sqlite, mysql
- Object Storage: minio, Ceph
- 不同缓存淘汰机制
- ....

## 实现细节

除了最重要的 set 和 get 分别对应 硬盘的随机读写，还需要考虑如何实现 contains, (LRU) evictor, recover。

### filesystem

性能测试

```commandline
python -m benchmarks.persistence_kv_cache.test_filesystem_server
```


|                                     | memory-set(deferred=False) | memory-set(deferred=True) | memory-get | memory-stream_get | filesystem-set(deferred=False) | filesystem-set(deferred=True) | filesystem-get | filesystem-stream_get |
|-------------------------------------|----------------------------|---------------------------|------------|-------------------|--------------------------------|-------------------------------|----------------|-----------------------|
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | 0.586                      | 0.406                     | 0.893      | 0.346             | 1.086                          | 0.340                         | 1.148          | 0.462                 |
| Qwen/Qwen2.5-7B-Instruct            | 0.113                      | 0.081                     | 0.228      | 0.074             | 0.252                          | 0.083                         | 0.261          | 0.143                 |
| Qwen/Qwen2.5-3B-Instruct            | 0.077                      | 0.055                     | 0.137      | 0.048             | 0.178                          | 0.059                         | 0.185          | 0.098                 |
| THUDM/glm-4-9b-chat-1m              | 0.154                      | 0.110                     | 0.274      | 0.078             | 0.345                          | 0.115                         | 0.370          | 0.161                 |
| NousResearch/Hermes-3-Llama-3.1-8B  | 0.270                      | 0.205                     | 0.427      | 0.106             | 0.569                          | 0.183                         | 0.556          | 0.227                 |

> 表2 filesystem 对比 内存 性能甚至差距不大， amazing. set(deferred=False) 两倍差距， stream_get 两倍差距

- set 
  - 同一个文件夹下文件数量太多可能影响访问性能，可以通过增加两层文件夹方式减少同文件夹下文件数量。
  - 总文件数太多会不会影响性能还需要进一步测试
- get
  - 每次打开文件，linux 文件系统自动更新 access time，可以用 access time 实现 LRU，非常方便
- contains
  - 无论是每个block都去查文件是否存在，还是读取所有文件，批量查block是否存在都很慢，这就有两种实现方法
    - 在内存中维护存在的 block，如果block数量巨大，占用很多cpu内存
    - 使用额外的数据库维护 metadata，查询metadata有overhead，可能导致整体系统性能下降，甚至会出现额外数据库是用内存维护metadata的套娃行为
  - 使用 filesystem 作为轻量级的实现，先使用内存中维护存在的 block
- evictor
  - 使用文件系统 access time 实现 LRU
- 正常 recover
  - 读取文件夹下所有文件，和对应的 access time， 按 access time 升序写入 allocator
- 异常 recover
  - 文件数增多，频繁修改，会不会导致整个文件系统不可用的概率增加？
  - 因为 kv cache 丢数据影响不是很大，重新算一次就可以了，所以不需要引入 WAL (Write-ahead logging)， 节省一些读写
  - 每个block都是单独的文件，读写异常不会扩散，最好的方法是使用纠错码，确保数据的完整性和可靠性，如果读取时发现错误就可以把这块数据丢弃
  - 查查文件系统和硬件是否支持自动纠错码，避免套娃实现

总体来说 filesystem 实现 persistence kv cache 简单高效，当scale的时候才会有以下的缺点：
1. 文件数量太多，频繁修改，可能导致性能下降，以及整个文件系统不可用的概率增加
2. metadata 在内存中维护可能占用过多内存

### mmap

可以将多个 block 写入 一个文件，称之为 super-block。能解决文件太多的问题，但也会引入很多新问题。

- set 
  - 当数据库还没有满的时候，追加写，性能不错
  - 当数据库满了，需要开始淘汰block时，就有两种选择
    - 继续在新super-block追加写，然后想办法把淘汰的block删除，把空间释放出来，导致需要读写一整个旧super-block
    - 使用mmap更新旧super-block，如果发生异常，可能会导致一整个旧super-block出错，异常会扩散
    - 当然也可以写满了，这个机器就只读，cdn就是用这种方式，简单有效
- get
  - 随机读取
- contains
  - 没办法只使用内存维护，需要额外持久化 metadata 数据库维护存在的 block
- evictor
  - 没办法只使用内存维护，需要额外持久化 metadata 数据库维护 access time 实现 LRU，或者其他更复杂的 evictor 算法
- recover
  实际上就是recover metadata数据库

总体来说 mmap 需要再找一个 metadata数据库 做配合

## 未完待续

# bybrid

最期待的混合 cpu内存和 ssd 的 persistence kv cache 来了

```commandline
python -m benchmarks.persistence_kv_cache.test_bybrid_server
```


|                                     | memory-set(deferred=True) | filesystem-set(deferred=True) | bybrid-set(deferred=True) | memory-stream_get | filesystem-stream_get | bybrid-stream_get | 
|-------------------------------------|---------------------------|-------------------------------|---------------------------|-------------------|-----------------------|-------------------|
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | 0.399                     | 0.338                         | 0.402                     | 0.348             | 0.457                 | 0.356             | 
| Qwen/Qwen2.5-7B-Instruct            | 0.081                     | 0.084                         | 0.098                     | 0.064             | 0.126                 | 0.061             | 
| Qwen/Qwen2.5-3B-Instruct            | 0.056                     | 0.059                         | 0.069                     | 0.054             | 0.101                 | 0.048             | 
| THUDM/glm-4-9b-chat-1m              | 0.109                     | 0.117                         | 0.130                     | 0.076             | 0.166                 | 0.085             | 
| NousResearch/Hermes-3-Llama-3.1-8B  | 0.208                     | 0.170                         | 0.231                     | 0.112             | 0.233                 | 0.120             | 

性能略高于 filesystem 略低于 memory。 符合预期