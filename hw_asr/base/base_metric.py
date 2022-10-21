import multiprocessing
from abc import ABC
from typing import List

import numpy as np
import torch

from hw_asr.text_encoder import CTCLMCharTextEncoder


class BaseMetric(ABC):
    def __init__(self, name=None, train=False, eval=False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.train = train
        self.eval = eval

    def __call__(self, **batch):
        raise NotImplementedError()


class BaseTextMetric(BaseMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_predicted_text(self, **batch) -> List[str]:
        raise NotImplementedError()

    def calc_metric(self, predicted_text: str, target_text: str) -> float:
        raise NotImplementedError()

    def __call__(self, **batch):
        predicted_texts = self.process_predicted_text(**batch)
        target_texts = batch["text"]
        metrics = [
            self.calc_metric(predicted_text, target_text)
            for predicted_text, target_text in zip(predicted_texts, target_texts)
        ]
        return np.mean(metrics)


class BaseLMMetric(BaseTextMetric, ABC):
    def __init__(
            self,
            text_encoder: CTCLMCharTextEncoder,
            n_jobs: int = 1,
            beam_size: int = 10,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decoder = text_encoder.decoder
        self.n_jobs = n_jobs
        self.beam_size = beam_size

    def process_predicted_text(
            self,
            log_probs: torch.Tensor,
            log_probs_length: torch.Tensor,
            **kwargs
    ) -> List[str]:
        log_probs = log_probs.detach().cpu().numpy()
        log_probs_length = log_probs_length.detach().cpu().numpy()

        truncated_log_probs = [
            log_prob[:length]
            for log_prob, length in zip(log_probs, log_probs_length)
        ]

        beam_size = self.beam_size
        with multiprocessing.get_context("fork").Pool(self.n_jobs) as pool:
            predicted_texts = self.decoder.decode_batch(pool, truncated_log_probs, beam_width=beam_size)

        return predicted_texts
