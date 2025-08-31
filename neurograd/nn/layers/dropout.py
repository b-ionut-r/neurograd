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
            dtype = X.data.dtype
            rp = xp.asarray(keep_prob, dtype=dtype)
            mask = (xp.random.rand(*X.shape).astype(dtype) < rp).astype(dtype)
            mask = mask / rp
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
            dtype = X.data.dtype
            rp = xp.asarray(keep_prob, dtype=dtype)
            mask = (xp.random.rand(X.shape[0], X.shape[1], 1, 1).astype(dtype) < rp).astype(dtype)
            mask = mask / rp
            mask = Tensor(mask, requires_grad=False)
            X = X * mask
        return X

