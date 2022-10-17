class BaseMetric:
    def __init__(self, name=None, train=False, eval=False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.train = train
        self.eval = eval

    def __call__(self, **batch):
        raise NotImplementedError()
