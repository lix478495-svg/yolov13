import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .block import C2f, C3
from .conv import Conv


class SkaFn(Function):
    """Spatial kernel aggregation with explicit autograd for dynamic depthwise-like weighting."""

    @staticmethod
    @custom_fwd
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2
        n, ic, h, width = x.shape
        _, wc, _, _, _ = w.shape
        groups = ic // wc

        ctx.ks = ks
        ctx.pad = pad
        ctx.groups = groups
        ctx.save_for_backward(x, w)

        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        x_windows = x_padded.unfold(2, ks, 1).unfold(3, ks, 1)
        x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous().view(n, ic, ks * ks, h, width)

        x_grouped = x_windows.view(n, groups, wc, ks * ks, h, width)
        w_grouped = w.view(n, 1, wc, ks * ks, h, width)
        out_grouped = torch.sum(x_grouped * w_grouped, dim=3)
        return out_grouped.view(n, ic, h, width)

    @staticmethod
    @custom_bwd
    def backward(ctx, go: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ks = ctx.ks
        pad = ctx.pad
        groups = ctx.groups
        x, w = ctx.saved_tensors
        n, ic, h, width = x.shape
        _, wc, k_sq, w_h, w_w = w.shape

        gx = None
        if ctx.needs_input_grad[0]:
            go_padded = F.pad(go, (pad, pad, pad, pad), mode="constant", value=0.0)
            go_windows = go_padded.unfold(2, ks, 1).unfold(3, ks, 1)
            go_windows = go_windows.permute(0, 1, 4, 5, 2, 3).contiguous().view(n, ic, ks * ks, h, width)

            go_grouped = go_windows.view(n, groups, wc, ks * ks, h, width)
            w_grouped = w.view(n, 1, wc, ks * ks, h, width)
            gx_grouped = torch.sum(go_grouped * w_grouped, dim=3)
            gx = gx_grouped.view(n, ic, h, width)

        gw = None
        if ctx.needs_input_grad[1]:
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
            x_windows = x_padded.unfold(2, ks, 1).unfold(3, ks, 1)
            x_windows = x_windows.permute(0, 1, 4, 5, 2, 3).contiguous().view(n, ic, ks * ks, h, width)

            x_grouped = x_windows.view(n, groups, wc, ks * ks, h, width)
            go_grouped = go.view(n, groups, wc, 1, h, width)
            gw = (x_grouped * go_grouped).sum(dim=1)
            if gw.shape != w.shape:
                gw = gw[:, :wc, :k_sq, :w_h, :w_w].contiguous()

        return gx, gw


class SKA(nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w)


class Conv2d_BN(nn.Sequential):
    def __init__(self, c1, c2, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("conv", nn.Conv2d(c1, c2, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(c2))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class LKP(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        if dim % groups != 0:
            raise ValueError(f"LKP expects dim % groups == 0, but got dim={dim}, groups={groups}.")
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks**2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks**2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim
        nn.init.zeros_(self.cv4.weight)
        nn.init.zeros_(self.cv4.bias)

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        return w.view(b, self.dim // self.groups, self.sks**2, h, width)


class LSConv(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8):
        super().__init__()
        self.lkp = LKP(dim, lks=lks, sks=sks, groups=groups)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        return x + self.gamma * self.bn(self.ska(x, self.lkp(x)))


class Bottleneck_LSConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LSConv(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k_LSConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_LSConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_LSConv(C2f):
    """C3k2 variant with LSConv bottlenecks for stronger local small-object feature aggregation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_LSConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_LSConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
