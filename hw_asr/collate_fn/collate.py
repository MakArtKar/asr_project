import logging
from typing import List

import torch
from torch.utils.data import default_collate


logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    padded_items = []
    keys_to_pad = ("audio", "spectrogram", "text_encoded")
    keys_to_squeeze = ("spectrogram", "text_encoded")

    lengths = {}
    for key in keys_to_pad:
        lengths[key] = max((item[key].shape[-1] for item in dataset_items))

    for item in dataset_items:
        result_item = {}
        for key, value in item.items():
            if key in keys_to_pad:
                pad_width = (0, lengths[key] - item[key].shape[-1])
                result_item[f"{key}_length"] = value.shape[-1]
                value = torch.nn.functional.pad(value, pad_width, "constant", 0)
            if key in keys_to_squeeze:
                value = value.squeeze(0)
            result_item[key] = value
        padded_items.append(result_item)

    result = default_collate(padded_items)
    return result
