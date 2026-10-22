"""Concentration nulls for the head partition. A native head is the model's own 128-dim slice of the pre-o_proj space; the nulls below are
alternative, function-preserving coordinate decompositions of the same update (rotated 128-dim blocks, permuted-coordinate blocks) plus a head-shuffled
alignment null. They are NOT independent computational heads."""
import torch

from causal_subspaces.rotations.haar import haar_orthogonal, two_frame_pair


def block_sums(x: torch.Tensor, hd: int) -> torch.Tensor:
    """(..., H*HD) -> (..., H) block sums."""
    return x.reshape(*x.shape[:-1], -1, hd).sum(-1)


def native_head_delta(delta: torch.Tensor, gain: torch.Tensor, hd: int) -> torch.Tensor:
    return block_sums(delta * gain, hd)


def haar_block_delta(delta, gain, hd, seed, device="cuda"):
    R = haar_orthogonal(delta.numel(), seed, device)
    return block_sums((R @ delta) * (R @ gain), hd)


def haar_block_delta_sampler(delta, gain, hd, n, seed):
    Rd, Rg = two_frame_pair(delta, gain, n, seed)
    return block_sums(Rd * Rg, hd)


def permuted_block_delta(delta, gain, hd, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(delta.numel(), generator=g).to(delta.device)
    return block_sums(delta[perm] * gain[perm], hd)


def head_shuffled_delta(delta, gain, hd, seed):
    """Permute the per-head gain slices among the heads of the layer (alignment destroyed, per-head marginals kept)."""
    H = delta.numel() // hd
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(H, generator=g).to(delta.device)
    return block_sums(delta * gain.view(H, hd)[perm].reshape(-1), hd)


def match_layer_mass(null_delta: torch.Tensor, native_mass: float) -> torch.Tensor:
    """Rescale a null per-layer head vector so its total |Delta| equals the native layer's (layer-matched pooling)."""
    m = null_delta.abs().sum(-1, keepdim=True)
    return null_delta * (native_mass / m)
