from wde.tasks.decode_only.output_last_hidden_states.arg_utils import \
    DecodeOnlyOutputLastHiddenStatesEngineArgs


class RetrieverDecodeOnlyEngineArgs(DecodeOnlyOutputLastHiddenStatesEngineArgs
                                    ):

    def __post_init__(self):
        super().__post_init__()
        self.output_last_hidden_states = True
