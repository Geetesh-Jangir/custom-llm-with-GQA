from __future__ import annotations
import torch
import math


class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim"""

    def __init__(
        self,
        head_dim: int,
        max_pos: int,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        assert head_dim % 2 == 0, "RoPE head_dim must be even "
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)

    def get(self, position):
        # position: (T,) or (1,T)
        if position.dim() == 2:
            position = position[0]
        need = int(position.max().item()) + 1 if position.numel() > 0 else 1
        if need > self.max_pos:
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[position]
        sin = self.sin[position]
        return cos, sin

    def _build(self, max_pos: int):
        self.max_pos = max_pos
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device).float()
                / self.head_dim
            )
        )
        t = torch.arange(max_pos, device=self.device).float()
        freqs = torch.outer(
            t, inv_freq
        )  # torch.outer does outer product and dimension will be (max_pos,head_dim//2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)


def apply_rope_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Rotate pairs along last dim of RoPE.
    x:(B,H,T,D) with D even; cos/sin : (T,D/2)
    """

    assert x.size(-1) % 2 == 0
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out
