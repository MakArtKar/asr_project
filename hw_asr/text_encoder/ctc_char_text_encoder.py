from collections import defaultdict
from typing import List, NamedTuple, Iterator, Tuple

import numpy as np
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
        self.ind2char: dict[int, str] = dict(enumerate(vocab))
        self.char2ind: dict[str, int] = {v: k for k, v in self.ind2char.items()}

    def ctc_append(self, text: List[str], last_char: str, new_char: str):
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
    def get_prob(prefix_prob, new_char_prob) -> float:
        return prefix_prob * new_char_prob

    @staticmethod
    def merge_hypos(hypos: List[Tuple[List[str], str, float]]) -> List[Tuple[List[str], str, float]]:
        dict_hypos = defaultdict(float)
        for text, last_char, prob in hypos:
            dict_hypos[(''.join(text), last_char)] += prob
        hypos = []
        for (text, last_char), prob in dict_hypos.items():
            hypos.append((list(text), last_char, prob))
        return hypos

    def extend_text_by_char(self, text: List[str], last_char: str, prob: float, char_probs: torch.Tensor) -> \
            Iterator[Tuple[List[str], str, float]]:
        orig_text_length = len(text)
        for new_char, new_char_prob in zip(self.char2ind, char_probs):
            new_text = self.ctc_append(text, last_char, new_char)
            text = text[:orig_text_length]
            yield (
                new_text,
                new_char,
                self.get_prob(prob, new_char_prob)
            )

    @staticmethod
    def cut_hypos_number(hypos: List[Tuple[List[str], str, float]], beam_size: int) -> List[Tuple[List[str], str, float]]:
        hypos = list(sorted(hypos, key=lambda x: x[-1], reverse=True))[:beam_size]
        return hypos

    def beam_step(self, hypos: List[Tuple[List[str], str, float]], char_probs: torch.Tensor, beam_size: int):
        new_hypos = []
        for text, last_char, prob in hypos:
            new_hypos.extend(self.extend_text_by_char(text, last_char, prob, char_probs))

        new_hypos = self.merge_hypos(new_hypos)
        new_hypos = self.cut_hypos_number(new_hypos, beam_size)

        return new_hypos

    def ctc_beam_search(self, log_probs: np.ndarray, probs_length: int,
                        beam_size: int = 10) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        probs = np.exp(log_probs)[:probs_length]
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = [(list(""), self.EMPTY_TOK, 1.)]
        for char_probs in probs:
            hypos = self.beam_step(hypos, char_probs, beam_size)
        hypos = [(text, '', prob) for text, _, prob in hypos]
        hypos = self.merge_hypos(hypos)
        hypos = self.cut_hypos_number(hypos, beam_size)
        hypos = [Hypothesis(text=''.join(text), prob=prob) for text, _, prob in hypos]
        return hypos

