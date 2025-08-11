import os
import numpy as np
import neurograd as ng
from neurograd.nn.layers import Linear


def test_module_state_dict_roundtrip():
    layer = Linear(4, 3, use_bias=True)
    # Capture original weights/bias via state_dict (should be copies)
    sd = layer.state_dict()
    assert "weight" in sd and ("bias" in sd or not layer.use_bias)
    w0 = sd["weight"].copy()
    b0 = sd.get("bias", None)

    # Mutate parameters
    layer.weight.data[...] = 0
    if layer.use_bias:
        layer.bias.data[...] = 0

    # Load back
    out = layer.load_state_dict(sd, strict=True)
    assert out["missing_keys"] == []
    assert out["unexpected_keys"] == []

    # Validate restoration
    assert np.allclose(ng.xp.asnumpy(layer.weight.data) if hasattr(ng.xp, "asnumpy") else layer.weight.data, w0)
    if layer.use_bias and b0 is not None:
        restored_b = ng.xp.asnumpy(layer.bias.data) if hasattr(ng.xp, "asnumpy") else layer.bias.data
        assert np.allclose(restored_b, b0)


def test_ng_save_creates_file(tmp_path):
    layer = Linear(2, 2)
    path = tmp_path / "model.pkl"
    ng.save(layer, str(path))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

