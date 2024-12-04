from typing import Optional

import torch
import torch.nn as nn

from wde.workflows.core.backends.vocab_embedding import VocabParallelEmbedding

from .sampling_metadata import SamplingMetadata


class LogitsProcessor(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:

        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.logits_as_input:
            logits = hidden_states
        else:
            hidden_states = _prune_hidden_states(hidden_states,
                                                 sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)

        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

        return logits

    def _get_logits(self, hidden_states: torch.Tensor,
                    lm_head: VocabParallelEmbedding,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = lm_head.linear_method.apply(lm_head,
                                             hidden_states,
                                             bias=embedding_bias)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    return hidden_states.index_select(0,
                                      sampling_metadata.selected_token_indices)
