from ..module import Module

class Dropout(Module):
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        super().__init__()
    def forward(self, X):
        import neurograd as ng
        from neurograd import xp, Tensor
        if self.training:
            keep_prob = 1 - self.dropout_rate
            mask = (xp.random.rand(*X.shape) < keep_prob) / keep_prob
            mask = Tensor(mask, requires_grad=False)
            X = X * mask
        return X
    

class Dropout2D(Module):
    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        super().__init__()
    def forward(self, X):
        import neurograd as ng
        from neurograd import xp, Tensor
        if self.training:
            keep_prob = 1 - self.dropout_rate
            # For NCHW format: (N, C, H, W), apply dropout per channel
            mask = (xp.random.rand(X.shape[0], X.shape[1], 1, 1) < keep_prob) / keep_prob
            mask = Tensor(mask, requires_grad=False)
            X = X * mask
        return X

