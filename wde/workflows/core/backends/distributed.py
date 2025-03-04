from typing import List, Optional

import torch


def get_pp_group():
    return _PP


def get_tensor_model_parallel_world_size():
    return 1


def get_tensor_model_parallel_rank():
    return 0


class FakeGroupCoordinator:
    rank: int = 0
    ranks: List[int] = [0]
    world_size: int = 1
    local_rank: int = 0
    rank_in_group: int = 0

    def destroy(self):
        pass

    @property
    def first_rank(self):
        return self.ranks[0]

    @property
    def last_rank(self):
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        return self.rank == self.last_rank

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        return input_


_PP: Optional = FakeGroupCoordinator()


def patch_parallel_state():
    import vllm.config as config
    import vllm.distributed.parallel_state as parallel_state

    parallel_state._WORLD = FakeGroupCoordinator()
    parallel_state._TP = FakeGroupCoordinator()
    parallel_state._PP = FakeGroupCoordinator()

    config._current_vllm_config = config.VllmConfig()