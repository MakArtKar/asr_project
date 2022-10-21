import math

import torchaudio
from datasets import load_dataset

from hw_asr.base.base_dataset import BaseDataset


class CommonVoice(BaseDataset):
    def __init__(self, part, limit=None, *args, **kwargs):
        torchaudio.set_audio_backend("sox_io")
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            use_auth_token=True
        )[part]
        super().__init__([], limit=limit, *args, **kwargs)
        if limit is not None:
            self.dataset = self.dataset.shuffle().shard(
                num_shards=math.ceil(len(self.dataset) / limit),
                index=0,
            )

    def __getitem__(self, ind):
        data_dict = self.dataset[ind]
        audio_path = data_dict["audio"]["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "text": data_dict["sentence"],
            "text_encoded": self.text_encoder.encode(data_dict["sentence"]),
            "audio_path": audio_path,
        }

    def __len__(self):
        return len(self.dataset)

