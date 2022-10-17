import unittest
from collections import defaultdict

import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder, Hypothesis


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()
        chars = [
            ['i', '^'],
            ['^', 'i'],
            ['^', ' '],
            [' ', '^'],
        ]
        probs = []
        for chars_step in chars:
            probs_step = [0.] * len(text_encoder.char2ind)
            for c in chars_step:
                ind = text_encoder.char2ind[c]
                probs_step[ind] = 1. / len(chars_step)
            probs.append(probs_step)
        probs = torch.tensor(probs)
        beam_search_output = text_encoder.ctc_beam_search(probs, len(probs), beam_size=100)

        result = defaultdict(float)
        for mask in range(2 ** len(chars)):
            text = []
            for i in range(len(chars)):
                text.append(chars[i][int(mask & (2 ** i) > 0)])
            text = text_encoder.ctc_decode([text_encoder.char2ind[c] for c in text])
            result[text] += 1 / 2 ** len(chars)
        result = [Hypothesis(text=text, prob=prob) for text, prob in result.items()]
        result = list(sorted(result, key=lambda x: (x.prob, x.text), reverse=True))
        beam_search_output = list(sorted(beam_search_output, key=lambda x: (x.prob, x.text), reverse=True))

        self.assertEqual(result, beam_search_output[:len(result)])
