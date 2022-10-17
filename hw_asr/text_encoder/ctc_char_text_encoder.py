from collections import defaultdict
from typing import List, NamedTuple, Iterator

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

    def ctc_append(self, text: list[str], last_char: str, new_char: str):
        if new_char != last_char and new_char != self.EMPTY_TOK:
            text.append(new_char)
        return text

    def ctc_decode(self, inds: List[int]) -> str:
        output = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            c = self.ind2char[ind]
            output = self.ctc_append(output, last_char, c)
            last_char = c
        return ''.join(output)

    @staticmethod
    def conditional_prob(prefix_prob, new_char_prob) -> float:
        return prefix_prob * new_char_prob

    @staticmethod
    def merge_hypos(hypos: list[tuple[list[str], str, float]]) -> list[tuple[list[str], str, float]]:
        dict_hypos = defaultdict(float)
        for text, last_char, prob in hypos:
            dict_hypos[(''.join(text), last_char)] += prob
        hypos = []
        for (text, last_char), prob in dict_hypos.items():
            hypos.append((list(text), last_char, prob))
        return hypos

    def extend_text(self, text: list[str], last_char: str, prob: float, char_probs: torch.Tensor) -> \
            Iterator[tuple[list[str], str, float]]:
        orig_text_length = len(text)
        for new_char, new_char_prob in zip(self.alphabet, char_probs):
            new_text = self.ctc_append(text, last_char, new_char)
            text = text[:orig_text_length]
            yield (
                new_text,
                new_char,
                self.conditional_prob(prob, new_char_prob)
            )

    @staticmethod
    def cut_hypos_number(hypos: list[tuple[list[str], str, float]], beam_size: int) -> list[tuple[list[str], str, float]]:
        hypos = list(sorted(hypos, key=lambda x: x[-1], reverse=True))[:beam_size]
        return hypos

    def beam_step(self, hypos: list[tuple[list[str], str, float]], char_probs: torch.Tensor, beam_size: int):
        new_hypos = []
        for text, last_char, prob in hypos:
            new_hypos.extend(self.extend_text(text, last_char, prob, char_probs))
        new_hypos = self.merge_hypos(new_hypos)
        new_hypos = self.cut_hypos_number(new_hypos, beam_size)
        return new_hypos

    def ctc_beam_search(self, probs: torch.Tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        probs = probs[:probs_length]
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = [(list(""), self.EMPTY_TOK, 1.)]
        for char_probs in probs:
            hypos = self.beam_step(hypos, char_probs, beam_size)
        hypos = [Hypothesis(text=''.join(text), prob=prob) for text, _, prob in hypos]
        return hypos

