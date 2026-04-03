import torch


class RMSNorm(torch.nn.Module):
    """
    Docstring for RMSNorm
    Root mean square layer normalization.
    y = x * g / rms(x), rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
