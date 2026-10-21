"""Orthogonal transforms of the post-gating MLP coordinates (Phase 1 nulls). Three DISTINCT families, never to be pooled:
  * haar_orthogonal      : true Haar R in O(m) (QR of a Gaussian matrix, sign-corrected).
  * two_frame_pair       : exact sampler of (R delta, R g) for Haar R, O(m) per draw (valid for statistics that only depend on that pair).
  * StructuredRotation   : block-Haar + random permutations, orthogonal but NOT Haar.
Rotating post-gating activations: a_rot = R a, W_rot = W_down R^T  =>  W_rot a_rot = W_down a (function preserving)."""
import torch


def haar_orthogonal(m: int, seed: int, device="cuda", dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(m, m, generator=g, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def two_frame_pair(delta: torch.Tensor, g: torch.Tensor, n: int, seed: int):
    """n draws of (R delta, R g) for R ~ Haar(O(m)). Uses: R delta_hat = u1, R g_perp_hat = u2 with (u1, u2) a Haar-random
    orthonormal 2-frame (QR of a 2-column Gaussian). Returns two (n, m) tensors."""
    dev, m = delta.device, delta.numel()
    delta, g = delta.double(), g.double()
    nd = delta.norm()
    dh = delta / nd
    par = g @ dh
    gp = g - par * dh
    ngp = gp.norm()
    gen = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, m, 2, generator=gen, device=dev, dtype=torch.float64)
    Q, R = torch.linalg.qr(X)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    u1, u2 = Q[..., 0], Q[..., 1]
    return (nd * u1).float(), (par * u1 + ngp * u2).float()


class StructuredRotation:
    """y = P_r B_r ... P_1 B_1 x : `rounds` of (random permutation, then independent Haar blocks of size `block`)."""

    def __init__(self, m: int, seed: int, block: int = 1024, rounds: int = 3, device="cuda"):
        assert m % block == 0
        self.m, self.block, self.nb, self.device = m, block, m // block, device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.perms = [torch.randperm(m, generator=g).to(device) for _ in range(rounds)]
        self.blocks = []
        for _ in range(rounds):
            A = torch.randn(self.nb, block, block, generator=g)
            Q, R = torch.linalg.qr(A)
            self.blocks.append((Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)).to(device))

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        shp = X.shape
        for perm, Q in zip(self.perms, self.blocks):
            X = X[..., perm].reshape(*shp[:-1], self.nb, self.block)
            X = torch.einsum("bji,...bi->...bj", Q.to(X.dtype), X).reshape(shp)
        return X

    def matrix(self) -> torch.Tensor:
        return self.apply(torch.eye(self.m, device=self.device)).T  # columns = images of basis vectors -> R


def rotated_class_difference(R_delta: torch.Tensor, R_g: torch.Tensor) -> torch.Tensor:
    """Class-difference contributions in rotated coordinates: Delta_rot_j = (R delta)_j (R g)_j (class means are linear in a)."""
    return R_delta * R_g
