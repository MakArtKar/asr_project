from typing import List

import numpy as np
import torch
from pyctcdecode import build_ctcdecoder

from hw_asr.text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.ctc_char_text_encoder import Hypothesis


class CTCLMCharTextEncoder(CTCCharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        chars = list(self.char2ind)
        chars[0] = ''
        self.decoder = build_ctcdecoder(
            chars,
            kenlm_model_path="./data/decoders/3-gram.pruned.1e-7.arpa",
            alpha=0.6,
            beta=0.0001,
        )

    def ctc_beam_search(self, log_probs: np.ndarray, probs_length: int,
                        beam_size: int = 20) -> List[Hypothesis]:
        probs = np.exp(log_probs)
        result = self.decoder.decode_beams(probs, beam_width=beam_size)
        return [Hypothesis(text=t[0], prob=np.exp(t[-1])) for t in result]
