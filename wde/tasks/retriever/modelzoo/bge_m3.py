# Adapted from
# https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/modeling.py
# FlagEmbedding is licensed under the MIT License.

from typing import List, Optional

import torch
from torch import nn

from wde.backends.attention import AttentionBackend, AttentionMetadata
from wde.backends.quantization import QuantizationConfig
from wde.tasks.core.schema.execute_io import IntermediateTensors
from wde.tasks.encode_only.modelzoo.xlm_roberta import (LoadWeightsMixin,
                                                        XLMRobertaConfig,
                                                        XLMRobertaModel)


class BGEM3Model(nn.Module, LoadWeightsMixin):
    _ignore_weights_keys = [
        "roberta.pooler.dense.weight", "roberta.pooler.dense.bias"
    ]

    prefix = "roberta."

    def __init__(self,
                 config: XLMRobertaConfig,
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
        self.roberta = XLMRobertaModel(config, attn_backend, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        assert kv_caches is None

        sequence_output = self.roberta(
            input_ids,
            positions,
            attn_metadata,
        )

        seq_start_loc = attn_metadata.seq_start_loc

        dense_vecs = sequence_output[seq_start_loc[:-1]]

        if self.normalized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)

        return dense_vecs
