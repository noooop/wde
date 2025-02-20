NAMESERVER_CLASS = "wde.microservices.framework.nameserver.server:ZeroNameServer"
MANAGER_CLASS = "wde.microservices.framework.zero_manager.server:ZeroManager"

INFERENCE_ENGINE_CLASS = "wde.engine.zero_engine:ZeroEngine"
ENTRYPOINT_ENGINE_CLASS = "wde.microservices.entrypoints.http_entrypoint:HttpEntrypoint"

REMOTE_KVCACHE_ENGINE_CLASS = "wde.workflows.decoding.kv_cache_server.server:ZeroRemoteKVCacheServer"
