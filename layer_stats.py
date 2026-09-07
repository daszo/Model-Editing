"""Running activation statistics for DOME's closed-form update.

DOME.py collects decoder activations over preserved-knowledge examples and needs
the C matrix used to precondition the update (paper Eq. 8, the K0 K0^T term that
keeps edits from disturbing existing docID mappings). This mirrors the
running-statistics utility from ROME (Meng et al., util/runningstats.py +
rome/layer_stats.py), reduced to the interface DOME.py actually uses:

    stat = CombinedStat(dim)
    stat.add(vector)                 # accumulate one d-dim activation
    stat.mean()                      # -> [dim]
    stat.mom2.covariance()           # -> [dim, dim]

`Covariance.covariance()` returns the *uncentered second moment* E[x xᵀ]. This is
the quantity ROME/MEMIT use as C when solving the constrained update (K0 K0^T ∝
E[k kᵀ]); it is symmetric positive semi-definite, so with the mom2_eps ridge that
DOME adds before Cholesky it is safely invertible. (If you instead want the
centered covariance, subtract torch.outer(mean, mean).)
"""
import torch


class Covariance:
    def __init__(self, dim, dtype=torch.float32):
        self.dim = dim
        self.dtype = dtype
        self.count = 0
        # Buffers are allocated lazily on the device of the first added vector,
        # so accumulation stays on-GPU when activations are on-GPU.
        self._sum = None
        self._outer = None

    def _init_buffers(self, device):
        self._sum = torch.zeros(self.dim, dtype=self.dtype, device=device)
        self._outer = torch.zeros(self.dim, self.dim, dtype=self.dtype, device=device)

    def add(self, x):
        x = x.detach().flatten().to(self.dtype)
        if x.numel() != self.dim:
            raise ValueError(
                f"Covariance expected vectors of dim {self.dim}, got {x.numel()}"
            )
        if self._sum is None:
            self._init_buffers(x.device)
        elif x.device != self._sum.device:
            x = x.to(self._sum.device)
        self.count += 1
        self._sum += x
        self._outer += torch.outer(x, x)

    def _n(self):
        return max(self.count, 1)

    def mean(self):
        if self._sum is None:
            return torch.zeros(self.dim, dtype=self.dtype)
        return self._sum / self._n()

    def second_moment(self):
        """Uncentered second moment E[x xᵀ]."""
        if self._outer is None:
            return torch.zeros(self.dim, self.dim, dtype=self.dtype)
        return self._outer / self._n()

    def covariance(self):
        # ROME-consistent C matrix for the closed-form update (see module docstring).
        return self.second_moment()

    def to(self, *args, **kwargs):
        if self._sum is not None:
            self._sum = self._sum.to(*args, **kwargs)
            self._outer = self._outer.to(*args, **kwargs)
        return self


class CombinedStat:
    """Container tracking the mean and second moment of streamed activations."""

    def __init__(self, dim, dtype=torch.float32):
        self.dim = dim
        self.mom2 = Covariance(dim, dtype=dtype)

    def add(self, x):
        self.mom2.add(x)

    def mean(self):
        return self.mom2.mean()

    def covariance(self):
        return self.mom2.covariance()

    def __len__(self):
        return self.mom2.count
