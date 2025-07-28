from ..module import ModuleMixin

class Dropout(ModuleMixin):
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        super().__init__()
    def forward(self, X):
        import neurograd as ng
        from neurograd import xp
        if self.training:
            keep_prob = 1 - self.dropout_rate
            mask = xp.random.rand(*X.shape) < keep_prob
            X = X * mask / keep_prob
        return X

