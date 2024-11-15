import copy
import enum
import math
from array import array
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Mapping, Optional, Tuple, Union

from wde.tasks.decoding.backends.sampling_params import SamplingParams

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = List[Optional[Dict[int, Logprob]]]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = List[Dict[int, Logprob]]


class SequenceStatus(enum.IntEnum):
    """Status of a sequence."""
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    # Note: anything after SWAPPED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status > SequenceStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


class SequenceData:
    """Data associated with a sequence.

    Args:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output. Set to an empty list if
            None.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
        output_token_ids: Optional[List[int]] = None,
    ) -> None:
        self._prompt_token_ids = array('l', prompt_token_ids)
        self._prompt_token_ids_tuple: Tuple[int, ...] = tuple(prompt_token_ids)
        self._output_token_ids = array(
            'l', output_token_ids if output_token_ids is not None else [])

        self.cumulative_logprob = 0.0
        # The number of tokens that are computed (that run against the model).
        self._num_computed_tokens = 0
        self._stage: SequenceStage = SequenceStage.PREFILL

        self._update_cached_all_tokens()

    @staticmethod
    def from_token_counts(*token_counts: Tuple[int, int]) -> "SequenceData":
        if len(token_counts) == 0:
            return SequenceData.from_seqs([])

        arrs = [
            array(VLLM_TOKEN_ID_ARRAY_TYPE, [token_id]) * count
            for token_id, count in token_counts
        ]

        return SequenceData(reduce(array.__add__, arrs))

    def _update_cached_all_tokens(self):
        self._cached_all_token_ids: List[int] = list(self._prompt_token_ids +
                                                     self._output_token_ids)

    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        self._prompt_token_ids = array('l', new_prompt_token_ids)
        self._prompt_token_ids_tuple = tuple(new_prompt_token_ids)
        self._update_cached_all_tokens()

    @property
    def prompt_token_ids_array(self) -> array:
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self, new_output_token_ids) -> None:
        self._output_token_ids = array('l', new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_token_ids_array(self) -> array:
        return self._output_token_ids

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self._output_token_ids.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self._cached_all_token_ids

    def get_prefix_token_ids(
            self, num_tokens: int
    ) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Get prefix tokens, and make the return value hashable"""
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple,
                    tuple(self._output_token_ids[:num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (
            self._num_computed_tokens, self.get_len())
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL

    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.output_token_ids

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self._prompt_token_ids}, "
                f"output_token_ids={self._output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        input: The input of the sequence.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        input: "PromptInputs",
        block_size: int,
        eos_token_id: Optional[int] = None,
    ) -> None:
        self.seq_id = seq_id
        self.input = input
        self.block_size = block_size
        self.eos_token_id = eos_token_id

        self.data = SequenceData(self.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    @property
    def n_blocks(self) -> int:
        return math.ceil(self.get_len() / self.block_size)

    @property
    def prompt(self) -> Optional[str]:
        return self.input.prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.input.prompt_token_ids

    def get_output_text_to_return(self, buffer_length: int):
        # We return the full output text if the sequence is finished.
        truncate = buffer_length and not self.is_finished()
        return self.output_text[:-buffer_length] if truncate else (
            self.output_text)

    def hash_of_block(self, logical_idx: int) -> int:
        # TODO This can produce incorrect hash when block size > prompt size

        # Compute the number of tokens in the sequence
        # TODO: The current hashing function is O(L^2). We should optimize
        # this in the future.
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        hashed_tokens = self.data.get_prefix_token_ids(num_tokens)
        return hash(hashed_tokens)

    def num_hashed_tokens_of_block(self, logical_idx: int):
        return logical_idx * self.block_size + self.block_size

    def reset_state_for_recompute(self):
        """Reset the sequence states for recomputation."""
        self.data.reset_state_for_recompute()

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, Logprob],
    ) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 1.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if self.data.stage == SequenceStage.DECODE:
            return 1
        return self.data.get_num_uncomputed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == SequenceStage.PREFILL

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={self.n_blocks}, ")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        embeddings: The embeddings vectors of the prompt of the sequence group
            for an embedding model.
        pooling_params: The pooling parameters used to generate the pooling
            for an embedding model.
        encoder_seq: Optional, the single encoder sequence. Should be None
                     unless you are working with an encoder/decoder model.
        trace_headers: OpenTelemetry trace headers.
        prompt_adapter_request: Prompt Adapter request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        sampling_params: Optional[SamplingParams] = None,
        embeddings: Optional[List[float]] = None,
        encoder_seq: Optional[Sequence] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.embeddings = embeddings
        self.encoder_seq = encoder_seq
        self.trace_headers = trace_headers

    @property
    def prompt(self) -> Optional[str]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return self.seqs[0].prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return self.seqs[0].prompt_token_ids

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""

        if (self.sampling_params and self.sampling_params.n > self.num_seqs()):
            # At prompt stage, the sequence group is not yet filled up
            # and only have one sequence running. However, in the
            # generation stage, we will have `n` sequences running.
            return self.sampling_params.n
        # At sampling stages, return the number of actual sequences
        # that are not finished yet.
        return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return self.seqs
        return [seq for seq in self.seqs if seq.status == status]

    def is_encoder_decoder(self) -> bool:
        return self.encoder_seq is not None

    def get_encoder_seq(self) -> Optional[Sequence]:
        return self.encoder_seq

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs if not seq.is_finished()]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs if seq.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        for seq in self.seqs:
            if not seq.is_finished():
                seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        for seq in self.seqs:
            if not seq.is_finished():
                num_uncomputed_tokens += seq.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        if status is None:
            return len(self.seqs)

        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq
        self.seqs.append(seq)

    def remove(self, seq_id: int) -> None:
        seq = self.seqs_dict.pop(seq_id, None)
        if seq is None:
            raise ValueError(f"Sequence {seq_id} not found.")
        self.seqs.remove(seq)

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.seqs)

    def is_prefill(self) -> bool:
        # Every sequence should be in the same stage.
        return self.seqs[0].is_prefill()

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `AttentionMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        do_sample: True if sampling is required. Sampling is not required when
            e.g., prefill is chunked, and the current iteration only computes
            query tokens for prefill, we don't need sampling.
        token_chunk_size: The number of tokens to be processed (per sequence).
            None if chunking is not required.
        encoder_seq_data: Optional sequence data for encoder prompt
                          (SequenceGroup.encoder_seq). Should be None
                          unless you are working with an encoder/decoder
                          model.
        cross_block_table: Optional cross-attention block table associated
                           with the encoder prompt
                           (SequenceGroup.encoder_seq). Should be None
                           unless you are working with an encoder/decoder
                           model.
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        do_sample: bool = True,
        token_chunk_size: Optional[int] = None,
        computed_block_nums: Optional[List[int]] = None,
        encoder_seq_data: Optional[SequenceData] = None,
        cross_block_table: Optional[List[int]] = None,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.computed_block_nums = computed_block_nums
        self.encoder_seq_data = encoder_seq_data
        self.cross_block_table = cross_block_table
        self._token_chunk_size = token_chunk_size
        self.do_sample = do_sample

        # The number of speculative tokens adopted in this request.
        # None means specuative decoding is not used.
        # Zero means speculative decoding is disabled for some reasons.
        # TODO: We should maintain this states out of the sequence group.
        self.num_speculative_tokens = None

        if self._token_chunk_size is None:
            if is_prompt:
                self._token_chunk_size = list(seq_data.values())[0].get_len()
            else:
                self._token_chunk_size = 1

    @property
    def token_chunk_size(self) -> int:
        """Return the number of tokens to be processed (chunk size)."""
        assert self._token_chunk_size is not None
        return self._token_chunk_size
