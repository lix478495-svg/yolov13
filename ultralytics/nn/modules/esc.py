import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Sequence, Literal

from .conv import Conv
from .block import C3k2, C3k

ATTN_TYPE = Literal["Naive", "SDPA", "Flex"]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]

    return bias_mod


def feat_to_win(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x,
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c",
        heads=heads,
        wh=window_size[0],
        ww=window_size[1],
        qkv=3,
    )


def win_to_feat(x: torch.Tensor, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x,
        "(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)",
        h=h_div,
        w=w_div,
        wh=window_size[0],
        ww=window_size[1],
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0),
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

        bs = x1.shape[0]
        dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)
        x1_ = rearrange(x1, "b c h w -> 1 (b c) h w")
        x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size // 2, groups=bs * self.pdim)
        x1_ = rearrange(x1_, "1 (b c) h w -> b c h w", b=bs, c=self.pdim)

        x1 = F.conv2d(x1, lk_filter.to(x1.dtype), stride=1, padding=self.lk_size // 2) + x1_

        x = torch.cat([x1, x2], dim=1)
        return x

    def extra_repr(self):
        return f"pdim={self.pdim}"


class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int, kernel_size: int = 13):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:
        x = self.plk(x, lk_filter)
        x = self.aggr(x)
        return x


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: float):
        super().__init__()
        hidden_dim = int(dim * exp_ratio)
        self.proj = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, kernel_size // 2, groups=hidden_dim)
        self.aggr = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        x = self.aggr(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_func=None,
        attn_type: ATTN_TYPE = "Naive",
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.dim = dim
        window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_type = attn_type
        self.attn_func = attn_func
        self.relative_position_bias = nn.Parameter(
            torch.randn(num_heads, (2 * window_size[0] - 1) * (2 * window_size[1] - 1), dtype=torch.float32) * 0.001
        )

        if self.attn_type == "Flex":
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        idxs_window = []
        for head in range(heads):
            for h in range(window_size ** 2):
                for w in range(window_size ** 2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        return torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)

    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = x.shape[2] // self.window_size[0], x.shape[3] // self.window_size[1]

        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_win(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == "Naive":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = self.attn_func(q, k, v, bias)
        else:
            raise NotImplementedError(f"Attention type {self.attn_type} is not supported in this integration.")

        out = win_to_feat(out, self.window_size, h_div, w_div)
        out = self.to_out(out.to(dtype)[:, :, :h, :w])
        return out

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


def _geo_ensemble(k):
    k = k.detach()
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    return (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8


class ESCBlock(nn.Module):
    def __init__(
        self,
        dim,
        conv_blocks=2,
        kernel_size=13,
        window_size=8,
        num_heads=8,
        exp_ratio=1.25,
        attn_func=attention,
        attn_type: ATTN_TYPE = "Naive",
        use_ln: bool = False,
    ):
        super().__init__()
        pdim = dim // 4

        self.ln_proj = LayerNorm(dim)
        self.proj = ConvFFN(dim, 3, 2.0)

        self.ln_attn = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type)

        self.lns = nn.ModuleList([LayerNorm(dim) if use_ln else nn.Identity() for _ in range(conv_blocks)])
        self.pconvs = nn.ModuleList([ConvAttnWrapper(dim, pdim, kernel_size) for _ in range(conv_blocks)])
        self.convffns = nn.ModuleList([ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)])

        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))
        geo_filter = _geo_ensemble(self.plk_filter)
        for ln, pconv, convffn in zip(self.lns, self.pconvs, self.convffns):
            x = x + pconv(convffn(ln(x)), geo_filter)
        x = self.conv_out(self.ln_out(x))
        return x + skip


class C3k_ESC(C3k):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
        k=3,
        window_size=8,
        num_heads=8,
    ):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(ESCBlock(c_, window_size=window_size, num_heads=num_heads) for _ in range(n)))


class C3k2_ESC(C3k2):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
        window_size=8,
        num_heads=8,
    ):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_ESC(self.c, self.c, 2, shortcut, g, 1.0, window_size=window_size, num_heads=num_heads)
            if c3k
            else ESCBlock(self.c, window_size=window_size, num_heads=num_heads)
            for _ in range(n)
        )
