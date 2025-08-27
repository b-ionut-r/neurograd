from __future__ import annotations

from ..module import Module
from neurograd import Tensor, xp
import numpy as np
import random


def _as_array(x):
    return x.data if isinstance(x, Tensor) else xp.array(x)


def _ensure_nchw(arr):
    single = False
    if arr.ndim == 3:  # C,H,W
        arr = arr[None, ...]
        single = True
    if arr.ndim != 4:
        raise ValueError("Expected CHW or NCHW tensor")
    return arr, single


def _stack(lst, axis=0):
    return xp.stack(lst, axis=axis)


def _bilinear_resize_chw(img_chw, out_h, out_w):
    C, H, W = img_chw.shape
    if H == out_h and W == out_w:
        return img_chw
    # Compute source coordinates
    ys = xp.linspace(0, H - 1, out_h)
    xs = xp.linspace(0, W - 1, out_w)
    y, x = xp.meshgrid(ys, xs, indexing='ij')
    y0 = xp.floor(y).astype(xp.int32)
    x0 = xp.floor(x).astype(xp.int32)
    y1 = xp.clip(y0 + 1, 0, H - 1)
    x1 = xp.clip(x0 + 1, 0, W - 1)
    wy = (y - y0)
    wx = (x - x0)
    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx
    out = xp.empty((C, out_h, out_w), dtype=img_chw.dtype)
    for c in range(C):
        I = img_chw[c]
        Ia = I[y0, x0]
        Ib = I[y0, x1]
        Ic = I[y1, x0]
        Id = I[y1, x1]
        out[c] = w00 * Ia + w01 * Ib + w10 * Ic + w11 * Id
    return out


def _affine_sample_chw(img_chw, M, out_h, out_w, border_value=0.0):
    C, H, W = img_chw.shape
    ys = xp.arange(out_h)
    xs = xp.arange(out_w)
    y, x = xp.meshgrid(ys, xs, indexing='ij')
    ones = xp.ones_like(x, dtype=img_chw.dtype)
    coords = xp.stack([x, y, ones], axis=0).reshape(3, -1)
    M = xp.asarray(M, dtype=img_chw.dtype)
    # Map output -> input
    src = M @ coords
    sx = src[0].reshape(out_h, out_w)
    sy = src[1].reshape(out_h, out_w)
    # Bilinear sample with border
    x0 = xp.floor(sx).astype(xp.int32)
    y0 = xp.floor(sy).astype(xp.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wa = (x1 - sx) * (y1 - sy)
    wb = (sx - x0) * (y1 - sy)
    wc = (x1 - sx) * (sy - y0)
    wd = (sx - x0) * (sy - y0)
    def sample(I, xx, yy):
        inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        xx_c = xp.clip(xx, 0, W - 1)
        yy_c = xp.clip(yy, 0, H - 1)
        vals = I[yy_c, xx_c]
        if isinstance(vals, np.ndarray) or hasattr(xp, 'asnumpy'):
            # Apply border where outside
            vals = vals * inside + border_value * (~inside)
        return vals
    out = xp.empty((C, out_h, out_w), dtype=img_chw.dtype)
    for c in range(C):
        I = img_chw[c]
        Ia = sample(I, x0, y0)
        Ib = sample(I, x1, y0)
        Ic = sample(I, x0, y1)
        Id = sample(I, x1, y1)
        out[c] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


class RandomTransform(Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p

    def should_apply(self) -> bool:
        return self.training and (random.random() < self.p)

    def forward(self, X: Tensor):
        if not self.should_apply():
            return X
        arr = _as_array(X)
        arr, single = _ensure_nchw(arr)
        out = []
        for i in range(arr.shape[0]):
            img = arr[i]
            img = self._apply_one(img)
            out.append(img)
        out = _stack(out, axis=0)
        if single:
            out = out[0]
        return Tensor(out, dtype=getattr(X, 'dtype', xp.float32))

    def _apply_one(self, img_chw):  # override
        return img_chw


class RandomHorizontalFlip(RandomTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def _apply_one(self, img_chw):
        return img_chw[:, :, ::-1]


class RandomVerticalFlip(RandomTransform):
    def __init__(self, p: float = 0.0):
        super().__init__(p=p)

    def _apply_one(self, img_chw):
        return img_chw[:, ::-1, :]


class RandomResizedCrop(RandomTransform):
    def __init__(self, out_h: int, out_w: int, scale=(0.5, 1.0), ratio=(3/4, 4/3), p: float = 1.0):
        super().__init__(p=p)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.scale = scale
        self.ratio = ratio

    def _apply_one(self, img_chw):
        C, H, W = img_chw.shape
        area = H * W
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = np.exp(random.uniform(*log_ratio))
            h = int(round(np.sqrt(target_area / aspect)))
            w = int(round(np.sqrt(target_area * aspect)))
            if 0 < h <= H and 0 < w <= W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                crop = img_chw[:, top:top+h, left:left+w]
                return _bilinear_resize_chw(crop, self.out_h, self.out_w)
        # Fallback center crop then resize
        in_ratio = W / H
        if in_ratio < self.ratio[0]:
            w = W
            h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            h = H
            w = int(round(h * self.ratio[1]))
        else:
            w, h = W, H
        top = max(0, (H - h) // 2)
        left = max(0, (W - w) // 2)
        crop = img_chw[:, top:top+h, left:left+w]
        return _bilinear_resize_chw(crop, self.out_h, self.out_w)


class ShiftScaleRotate(RandomTransform):
    def __init__(self, shift_limit=0.0, scale_limit=0.0, rotate_limit=0.0, value: float = 0.0, p: float = 1.0):
        super().__init__(p=p)
        self.shift_limit = float(shift_limit)
        self.scale_limit = float(scale_limit)
        self.rotate_limit = float(rotate_limit)
        self.value = value

    def _apply_one(self, img_chw):
        C, H, W = img_chw.shape
        angle = random.uniform(-self.rotate_limit, self.rotate_limit) * np.pi / 180.0
        scale = 1.0 + random.uniform(-self.scale_limit, self.scale_limit)
        tx = random.uniform(-self.shift_limit, self.shift_limit) * W
        ty = random.uniform(-self.shift_limit, self.shift_limit) * H
        cos_a = np.cos(angle); sin_a = np.sin(angle)
        # Build output->input affine matrix
        cx, cy = W * 0.5, H * 0.5
        # Translation to origin, rotate+scale, translate back, then shift
        M = xp.array([[ scale * cos_a,  scale * sin_a,  (1 - scale * cos_a) * cx - scale * sin_a * cy + tx],
                      [-scale * sin_a,  scale * cos_a,  scale * sin_a * cx + (1 - scale * cos_a) * cy + ty]], dtype=img_chw.dtype)
        return _affine_sample_chw(img_chw, M, H, W, border_value=self.value)


class ColorJitter(RandomTransform):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p: float = 1.0):
        super().__init__(p=p)
        self.b = brightness; self.c = contrast; self.s = saturation; self.h = hue

    def _rgb_to_hsv(self, img):
        # img: C,H,W in [0,1]
        r, g, b = img[0], img[1], img[2]
        mx = xp.maximum(xp.maximum(r, g), b)
        mn = xp.minimum(xp.minimum(r, g), b)
        v = mx
        d = mx - mn + 1e-12
        s = d / (mx + 1e-12)
        # Hue
        hr = ((g - b) / d) % 6.0
        hg = ((b - r) / d) + 2.0
        hb = ((r - g) / d) + 4.0
        h = xp.where(mx == r, hr, xp.where(mx == g, hg, hb)) / 6.0
        h = xp.where(d < 1e-12, 0.0, h)
        return xp.stack([h, s, v], axis=0)

    def _hsv_to_rgb(self, img):
        h, s, v = img[0], img[1], img[2]
        h = (h % 1.0) * 6.0
        i = xp.floor(h).astype(xp.int32)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        r = xp.where(i == 0, v, xp.where(i == 1, q, xp.where(i == 2, p, xp.where(i == 3, p, xp.where(i == 4, t, v)))))
        g = xp.where(i == 0, t, xp.where(i == 1, v, xp.where(i == 2, v, xp.where(i == 3, q, xp.where(i == 4, p, p)))))
        b = xp.where(i == 0, p, xp.where(i == 1, p, xp.where(i == 2, t, xp.where(i == 3, v, xp.where(i == 4, v, q)))))
        return xp.stack([r, g, b], axis=0)

    def _apply_one(self, img_chw):
        img = img_chw
        dtype = img.dtype
        # assume [0,1] if float, else scale accordingly
        if xp.issubdtype(dtype, xp.floating):
            x = img
        else:
            x = img.astype(xp.float32) / 255.0
        # brightness
        if self.b > 0:
            factor = random.uniform(max(0, 1 - self.b), 1 + self.b)
            x = x * factor
        # contrast
        if self.c > 0:
            mean = xp.mean(x, axis=(1, 2), keepdims=True)
            factor = random.uniform(max(0, 1 - self.c), 1 + self.c)
            x = (x - mean) * factor + mean
        # saturation/hue
        if self.s > 0 or self.h > 0:
            hsv = self._rgb_to_hsv(xp.clip(x, 0, 1))
            if self.s > 0:
                s_fac = random.uniform(max(0, 1 - self.s), 1 + self.s)
                hsv[1] = xp.clip(hsv[1] * s_fac, 0, 1)
            if self.h > 0:
                h_shift = random.uniform(-self.h, self.h) / 360.0
                hsv[0] = (hsv[0] + h_shift) % 1.0
            x = self._hsv_to_rgb(hsv)
        x = xp.clip(x, 0, 1)
        if xp.issubdtype(dtype, xp.floating):
            return x.astype(dtype, copy=False)
        else:
            return (x * 255.0).astype(dtype, copy=False)


class RandomGamma(RandomTransform):
    def __init__(self, gamma_limit=(80, 120), p: float = 1.0):
        super().__init__(p=p)
        self.gamma_limit = gamma_limit

    def _apply_one(self, img_chw):
        dtype = img_chw.dtype
        x = img_chw.astype(xp.float32) / (255.0 if not xp.issubdtype(dtype, xp.floating) else 1.0)
        g = random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        x = xp.power(x, g)
        x = xp.clip(x, 0, 1)
        if xp.issubdtype(dtype, xp.floating):
            return x.astype(dtype, copy=False)
        else:
            return (x * 255.0).astype(dtype, copy=False)


class RGBShift(RandomTransform):
    def __init__(self, r_shift=0, g_shift=0, b_shift=0, p: float = 1.0):
        super().__init__(p=p)
        self.rs = r_shift; self.gs = g_shift; self.bs = b_shift

    def _apply_one(self, img_chw):
        out = img_chw.copy()
        out[0] = xp.clip(out[0] + self.rs, 0, xp.iinfo(out.dtype).max if not xp.issubdtype(out.dtype, xp.floating) else 1.0)
        out[1] = xp.clip(out[1] + self.gs, 0, xp.iinfo(out.dtype).max if not xp.issubdtype(out.dtype, xp.floating) else 1.0)
        out[2] = xp.clip(out[2] + self.bs, 0, xp.iinfo(out.dtype).max if not xp.issubdtype(out.dtype, xp.floating) else 1.0)
        return out


class GaussianNoise(RandomTransform):
    def __init__(self, std=0.1, p: float = 1.0):
        super().__init__(p=p)
        self.std = std

    def _apply_one(self, img_chw):
        dtype = img_chw.dtype
        x = img_chw.astype(xp.float32) / (255.0 if not xp.issubdtype(dtype, xp.floating) else 1.0)
        noise = xp.random.normal(0.0, self.std, size=x.shape).astype(xp.float32)
        x = xp.clip(x + noise, 0, 1)
        if xp.issubdtype(dtype, xp.floating):
            return x.astype(dtype, copy=False)
        else:
            return (x * 255.0).astype(dtype, copy=False)


class CoarseDropout(RandomTransform):
    def __init__(self, num_holes=(1, 1), hole_h=(0.1, 0.1), hole_w=(0.1, 0.1), fill=0.0, p: float = 1.0):
        super().__init__(p=p)
        self.num_holes = num_holes
        self.hole_h = hole_h
        self.hole_w = hole_w
        self.fill = fill

    def _apply_one(self, img_chw):
        C, H, W = img_chw.shape
        n = random.randint(self.num_holes[0], self.num_holes[1])
        out = img_chw.copy()
        for _ in range(n):
            hh = int(H * random.uniform(self.hole_h[0], self.hole_h[1]))
            ww = int(W * random.uniform(self.hole_w[0], self.hole_w[1]))
            top = random.randint(0, max(0, H - hh))
            left = random.randint(0, max(0, W - ww))
            out[:, top:top+hh, left:left+ww] = self.fill
        return out


class ComposeAug(Module):
    def __init__(self, *transforms: Module):
        super().__init__()
        self.transforms = transforms
        for i, t in enumerate(transforms):
            self.add_module(f"t{i}", t)

    def forward(self, X: Tensor):
        for t in self.transforms:
            X = t(X)
        return X
