import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # CuPy not installed in many CI envs
    cp = None


def rand_img(xp, C=3, H=8, W=8, dtype=np.uint8):
    x = (xp.random.random((C, H, W)) * 255).astype(dtype)
    return x


def test_compose_order():
    from neurograd.transforms import Compose
    from neurograd.transforms.random import RandomHorizontalFlip

    class AddOne:
        def __call__(self, x):
            return x + 1

    class TimesTwo:
        def __call__(self, x):
            return x * 2

    x = np.zeros((1, 2, 2), dtype=np.float32)
    t = Compose([AddOne(), TimesTwo(), AddOne()])
    y = t(x)
    # ((0 + 1) * 2) + 1 = 3
    assert np.allclose(y, 3.0)


def test_random_flip_seed_reproducible():
    from neurograd.transforms.random import RandomHorizontalFlip

    x = rand_img(np)
    t1 = RandomHorizontalFlip(p=1.0, seed=42)
    t2 = RandomHorizontalFlip(p=1.0, seed=42)
    y1 = t1(x.copy())
    y2 = t2(x.copy())
    assert np.array_equal(y1, y2)


def test_random_crop_shape():
    from neurograd.transforms.random import RandomCrop

    x = rand_img(np, H=10, W=12)
    t = RandomCrop((6, 7), seed=0)
    y = t(x)
    assert y.shape == (x.shape[0], 6, 7)


def test_normalize_values():
    from neurograd.transforms import Normalize

    x = np.ones((3, 2, 2), dtype=np.float32)
    mean = (1.0, 2.0, 3.0)
    std = (1.0, 2.0, 4.0)
    t = Normalize(mean, std)
    y = t(x)
    expected = np.stack([
        (x[0] - 1.0) / 1.0,
        (x[1] - 2.0) / 2.0,
        (x[2] - 3.0) / 4.0,
    ], axis=0)
    assert np.allclose(y, expected)


def test_backend_parity_numpy_vs_cupy_if_available():
    if cp is None:
        return
    from neurograd.transforms.random import RandomCrop, RandomHorizontalFlip
    from neurograd.transforms import Normalize, Compose

    seed = 123
    x_np = rand_img(np, C=3, H=9, W=9, dtype=np.float32) / 255.0
    x_cp = cp.asarray(x_np)

    t_np = Compose([
        RandomCrop((5, 7), seed=seed),
        RandomHorizontalFlip(p=1.0, seed=seed),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    t_cp = Compose([
        RandomCrop((5, 7), seed=seed),
        RandomHorizontalFlip(p=1.0, seed=seed),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    y_np = t_np(x_np.copy())
    y_cp = t_cp(x_cp.copy())
    # Bring back to CPU for comparison
    assert np.allclose(cp.asnumpy(y_cp), y_np, atol=1e-6)

