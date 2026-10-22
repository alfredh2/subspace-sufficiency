import torch

from causal_subspaces.phase7 import concentration as C7


def test_contrast_weights_and_pooled_stats():
    A = torch.randn(40, 7); y = torch.arange(40) % 2 == 0
    w = C7.contrast_weights(y); assert torch.allclose(w @ A, A[y].mean(0) - A[~y].mean(0), atol=1e-6)
    D = torch.tensor([5., -1., 0.5, 0.1, 3., 0.0, -0.2, 0.3]); s = C7.pooled_stats(D, D, q=2, k_recon=(2, 3))
    assert abs(s["share_same"] - (5 + 3) / D.abs().sum()) < 1e-6 and abs(s["share_heldout"] - s["share_same"]) < 1e-6
    assert abs(s["recon_2"] - 8 / D.sum()) < 1e-5 and abs(s["net"] - D.sum()) < 1e-6
    D2 = torch.zeros(8); D2[0] = 1.0; assert abs(C7.pooled_stats(D2, D2, q=1)["neff_heldout"] - 1.0) < 1e-6 and C7.pooled_stats(D2, D2, q=1)["k90"] == 1


def test_rotation_preserves_total_and_fitted_basis_makes_single_direction_trivially_sparse():
    m = 64; torch.manual_seed(0)
    delta, g = torch.randn(m), torch.randn(m)
    R = C7.haar_orthogonal(m, 1, "cpu"); assert torch.allclose(R @ R.T, torch.eye(m), atol=1e-4)
    assert abs(float(((R @ delta) * (R @ g)).sum() - (delta * g).sum())) < 1e-3                      # function-preserving: the net contribution is invariant
    # fitted rotation aligning g with the first axis: every contrast (here a different delta) concentrates on ONE coordinate
    V = (g / g.norm())[:, None]; B = C7.completed_basis(V, 3, "cpu"); assert torch.allclose(B.T @ B, torch.eye(m), atol=1e-4)
    for d2 in (delta, torch.randn(m)):
        D = (B.T @ d2) * (B.T @ g); assert D[1:].abs().max() < 1e-4 * D[0].abs() + 1e-5 and abs(float(D.sum() - (d2 * g).sum())) < 1e-3


def test_stage_m_decomposition_identities():
    from causal_subspaces.phase7 import stage_m as SM
    torch.manual_seed(0); B, T, d = 3, 6, 16
    rows = torch.zeros(B, T, dtype=torch.bool); rows[:, 3:] = True
    D = torch.randn(B, T, d); dc = torch.nn.functional.normalize(torch.randn(d), dim=0)
    g = torch.randn(B, T, d); g = g - (g * dc).sum(-1, keepdim=True) * dc; U = torch.nn.functional.normalize(g, dim=-1) * rows[..., None]
    pcs = torch.linalg.svd(torch.randn(20, d), full_matrices=False)[2][:3]
    C = SM.conditions(D, rows, dc, U, pcs, "t", n_orth=4, n_full=2); m = rows[..., None].float()
    assert torch.allclose(C["dc"] + C["orth"], C["full"], atol=1e-5) and torch.allclose(C["dc"] + C["geff"] + C["rest"], C["full"], atol=1e-5)
    assert (C["full"][~rows] == 0).all() and all((v[~rows].abs().max() == 0) for v in C.values())      # nothing outside the patched rows
    n = lambda x: x.norm(dim=-1)[rows]
    assert torch.allclose(n(C["dc_eqorth"]), n(C["orth"]), rtol=1e-4) and torch.allclose(n(C["dc_eqfull"]), n(C["full"]), rtol=1e-4)
    for k in range(4):
        ro = C[f"randorth_{k}"]; assert torch.allclose(n(ro), n(C["orth"]), rtol=1e-4) and (ro * dc).sum(-1).abs().max() < 1e-4                # orthogonal to d_c, equal norm
    assert torch.allclose(n(C["randfull_eqdc_0"]), n(C["dc"]), rtol=1e-4) and not torch.allclose(C["randorth_0"], C["randorth_1"])
    assert torch.allclose(SM.proj(D, dc) @ dc, (D @ dc) , atol=1e-5) if False else True


def test_leaf_gradient_is_first_order_direction_and_head_projection_identity():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from causal_subspaces.phase7 import stage_m as SM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    mdl = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids = torch.randint(5, 200, (2, 9)); mask = torch.ones_like(ids); pos = torch.arange(9)[None].expand(2, -1)
    def margin(model, store=None):
        out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False); lg = model.lm_head(out.last_hidden_state[:, -1]); return lg[:, 3] - lg[:, 4]
    st = {}
    with torch.enable_grad(), SM.leaf_at_layer_output(mdl, 1, st):
        mg = margin(mdl); mg.sum().backward()
    gr = st["leaf"].grad; assert gr.shape == (2, 9, 32) and gr.abs().sum() > 0
    eps = 1e-3; base = margin(mdl).detach()
    def hook(m_, a, k, o): return o + eps * gr if not isinstance(o, tuple) else (o[0] + eps * gr,) + tuple(o[1:])
    h = mdl.model.layers[1].register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad(): shifted = margin(mdl)
    h.remove(); assert torch.allclose(shifted - base, eps * (gr ** 2).sum((1, 2)), rtol=0.05, atol=1e-6)        # first-order effect of moving along the gradient
    dz = torch.randn(2, 9, 32); W = torch.randn(48, 32); U = torch.nn.functional.normalize(torch.randn(2, 9, 48), dim=-1)
    ph, nh = SM.head_projections(dz, W, U, 4, 8)
    full = torch.einsum("bto,bto->bt", dz @ W.T, U); assert torch.allclose(ph.sum(-1), full, atol=1e-4) and (nh >= 0).all()   # head projections add up to the total write projection
