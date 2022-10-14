from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        output = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            c = self.ind2char[ind]
            if c != self.EMPTY_TOK and c != last_char:
                output.append(c)
            last_char = c

        return ''.join(output)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis(text='', prob=1.)]
        for prob in probs:
            new_hypos: dict[str, float] = defaultdict(int)
            for text, prefix_prob in hypos:
                for char, char_prob in zip(self.alphabet, prob):
                    last_char = text[-1] if text else self.EMPTY_TOK
                    if char == last_char:
                        new_hypos[text] += prefix_prob * char_prob
                    else:
                        new_hypos[text + last_char] += prefix_prob * char_prob
            hypos = [Hypothesis(text, prob) for text, prob in new_hypos.items()]
            hypos = list(sorted(hypos, key=lambda x: x.prob, reverse=True))[:beam_size]

        hypos = [Hypothesis(text.replace(self.EMPTY_TOK, ''), prob) for text, prob in hypos]
        return hypos
