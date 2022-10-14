from torchaudio.models import DeepSpeech

from hw_asr.model import BaselineModel


class TorchDeepSpeech(BaselineModel):
    def __init__(self, n_feats, n_class, n_hidden=512, dropout=0.1):
        super().__init__(n_feats, n_class, fc_hidden=n_hidden)
        self.net = DeepSpeech(n_feats, n_hidden, n_class, dropout)
