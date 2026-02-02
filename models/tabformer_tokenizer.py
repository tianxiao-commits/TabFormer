from transformers import PreTrainedTokenizer

class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):
        self._vocab = {}
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)

    def get_vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return len(self._vocab)