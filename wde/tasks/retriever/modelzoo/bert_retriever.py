# Adapted from
# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/flag_models.py
# FlagEmbedding is licensed under the MIT License.
# BertRetriever also supports Snowflake Arctic Embed (Family)
# Arctic is licensed under the Apache-2.

from typing import List, Optional

import torch
from torch import nn

from wde.backends.attention import AttentionBackend, AttentionMetadata
from wde.backends.quantization import QuantizationConfig
from wde.tasks.encode_only.modelzoo.bert import (BertConfig, BertModel,
                                                 LoadWeightsMixin)
from wde.tasks.retriever.schema.execute_io import RetrieverExecuteOutput
from wde.workflows.core.schema.execute_io import IntermediateTensors


class BertRetriever(nn.Module, LoadWeightsMixin):
    # bge v1.5 family
    # Snowflake Arctic Embed (Family)

    prefix = "bert."
    _ignore_weights_keys = [
        "bert.embeddings.position_ids", 'bert.pooler.dense.weight'
    ]

    def __init__(self,
                 config: BertConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None,
                 sentence_pooling_method="cls",
                 normalized=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.sentence_pooling_method = sentence_pooling_method
        assert self.sentence_pooling_method == 'cls'
        self.normalized = normalized

        self.bert = BertModel(config,
                              attn_backend,
                              quant_config=quant_config,
                              add_pooling_layer=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> RetrieverExecuteOutput:
        assert kv_caches is None

        sequence_output, pooled_output = self.bert(
            input_ids,
            positions,
            attn_metadata,
        )

        seq_start_loc = attn_metadata.seq_start_loc

        dense_vecs = sequence_output[seq_start_loc[:-1]]

        if self.normalized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)

        return RetrieverExecuteOutput(embeddings=dense_vecs)
