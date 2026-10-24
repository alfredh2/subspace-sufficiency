"""Phase 8 S1 confirm stage (run only AFTER discovery_freeze.json is committed). Retention (sufficiency) of the frozen carriers on fresh confirm items
(+ reused Stage 1b confirm items, flagged), and descriptive semantics/geometry/readout-likeness. No ablation, transplant, dose or compensation (B1)."""
import json, pickle
import numpy as np, torch
from sklearn.linear_model import LogisticRegression, Ridge
from causal_subspaces.experiment import get_reference
from causal_subspaces.models.hooks import capture_residual
from causal_subspaces.models.loading import encode, last_logits, load_model
from causal_subspaces.phase7.stage1 import run_stats, primary_accuracy_vec, cluster_boot, haar_subspace, texts_of, cand_array
from causal_subspaces.phase7.stage1b import build_1b, overlap, principal_cosines
from causal_subspaces.utils.config import ROOT, load_config

R = ROOT / "results/phase8/stage1"
C = R / "carriers"
fr = json.load(open(R / "discovery_freeze.json"))
cfg = load_config(ROOT / "configs/experiments/mvs_rank2_l22.yaml", [])
model, tok, rev = load_model(cfg["model"], "cuda")
d1b = build_1b("M", tok)["letters"]
fresh = pickle.load(open(R / "fresh_items.pkl", "rb"))
sets = {"fresh_conf": fresh["conf"]["items"], "reused_1b_conf": d1b["test"]}
train = d1b["train"]
rng = np.random.default_rng(0)
sub_train = [train[i] for i in rng.choice(len(train), 600, replace=False)]
W = model.lm_head.weight.detach().float()
norm_f = model.model.norm
cids = cand_array(train[:1])[0]


@torch.no_grad()
def final_pos_resid(items, h, bs=32):
    texts = texts_of(tok, items); order = sorted(range(len(items)), key=lambda i: len(texts[i]))
    out = torch.zeros(len(items), model.config.hidden_size)
    for s in range(0, len(order), bs):
        ii = order[s:s + bs]
        ids, mask, pos = encode(tok, [texts[i] for i in ii], "cuda")
        sink = []
        with capture_residual(model, h, sink):
            last_logits(model, ids, mask, pos)
        out[ii] = sink[0][:, -1].float().cpu()
    return out


def lens_acc(hv, gold):
    with torch.no_grad():
        lg = norm_f(hv.cuda().to(torch.bfloat16)).float() @ W[cids].T.cuda()
    return float((lg.argmax(1).cpu().numpy() == gold).mean()), lg.cpu().numpy()


def probe_acc(Xtr, ytr, Xte, yte):
    mu_, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu_) / sd, ytr)
    return float(clf.score((Xte - mu_) / sd, yte))


rel_of = lambda items: np.array([1 if "capital" in it.prompt.split("\n")[0] else 0 for it in items])
gold_of = lambda items: np.array([it.gold for it in items])
res = dict(frozen=str((R / "discovery_freeze.json").relative_to(ROOT)), hooks={})
for h in (25, 21):
    names = [n for n in fr["chains"] if n.startswith(f"h{h}_")]
    if not names:
        continue
    cfg["mvs"]["residual_index"] = h
    ref = get_reference(model, tok, cfg, rev, "cuda"); mu = ref["mu"].cuda()
    X_tr = final_pos_resid(sub_train, h) - ref["mu"].cpu().float()
    Xs = {k: final_pos_resid(v, h) - ref["mu"].cpu().float() for k, v in sets.items()}
    intact = {k: run_stats(model, tok, v) for k, v in sets.items()}
    acc_i = {k: float(primary_accuracy_vec(intact[k], v).mean()) for k, v in sets.items()}
    H = {}
    Bu = torch.linalg.qr(W[cids].T)[0]                          # (d, 4) basis of the four letter-unembedding rows
    for nm in names:
        s = int(nm.split("_s")[1]); K = fr["chains"][nm]["kstar_dev"]
        Us = [torch.load(R / "carriers" / f"h{h}_s{s}_k{k}.pt" if k > 1 else (ROOT / fr["chains"][nm]["carriers"]["1"]["file"])).cuda() for k in range(1, K + 1)]
        orth = max([float((Us[i].T @ Us[j]).abs().max()) for i in range(K) for j in range(i)] or [0.0])
        ch = dict(kstar_dev=K, max_pair_orth_err=orth, carriers={})
        coords = {}
        for k, U in enumerate(Us, 1):
            row = {}
            for sk, items in sets.items():
                st = run_stats(model, tok, items, h, "retain", U, mu, with_kl=True)
                acc = primary_accuracy_vec(st, items)
                groups = [it.group for it in items]
                kl = st["kl"]
                row[sk] = dict(kl=float(kl.mean()), kl_ci=cluster_boot(lambda ix: kl[ix].mean(), groups, 1000, k), acc=float(acc.mean()),
                               acc_ratio=float(acc.mean() / acc_i[sk]), sufficient=bool(kl.mean() <= 0.05 and acc.mean() >= 0.9 * acc_i[sk]),
                               ub_ok=bool(cluster_boot(lambda ix: kl[ix].mean(), groups, 1000, k)[1] <= 0.05),
                               dev_kl=fr["chains"][nm]["carriers"][str(k)]["dev_kl"],
                               logit_dev=(st["cand_logits"] - intact[sk]["cand_logits"]).tolist() if sk == "fresh_conf" else None)
            # descriptive semantics on fresh confirm (probe trained on train subsample, eval on confirm)
            Ut = U.cpu().float()
            ctr, cte = X_tr @ Ut, Xs["fresh_conf"] @ Ut
            coords[k] = (ctr, cte)
            ret = ref["mu"].cpu().float() + cte @ Ut.T                      # retained residual at final position
            la, _ = lens_acc(ret, gold_of(sets["fresh_conf"]))
            row["probe_letter_acc"] = probe_acc(ctr.numpy(), gold_of(sub_train), cte.numpy(), gold_of(sets["fresh_conf"]))
            row["probe_relation_acc"] = probe_acc(ctr.numpy(), rel_of(sub_train), cte.numpy(), rel_of(sets["fresh_conf"]))
            row["lens_acc_retained"] = la
            row["unembed_span_energy"] = float((Bu.T @ U.float()).pow(2).sum() / 4)
            ch["carriers"][k] = row
        ch["haar_unembed_span_energy_ref"] = 4 / 4096
        Hd, _ = lens_acc(Xs["fresh_conf"] + ref["mu"].cpu().float(), gold_of(sets["fresh_conf"]))
        ch["lens_acc_intact_residual"] = Hd
        # union of first min(K,4) carriers and random references
        q = min(K, 4)
        Uu = torch.cat([U.cpu().float() for U in Us[:q]], 1)
        ch["probe_union_first4"] = dict(letter=probe_acc((X_tr @ Uu).numpy(), gold_of(sub_train), (Xs["fresh_conf"] @ Uu).numpy(), gold_of(sets["fresh_conf"])),
                                         relation=probe_acc((X_tr @ Uu).numpy(), rel_of(sub_train), (Xs["fresh_conf"] @ Uu).numpy(), rel_of(sets["fresh_conf"])))
        ch["probe_full_residual"] = dict(letter=probe_acc(X_tr.numpy(), gold_of(sub_train), Xs["fresh_conf"].numpy(), gold_of(sets["fresh_conf"])))
        rnd = [haar_subspace(4096, 4, 900 + i).float().cpu() for i in range(5)]
        ch["probe_random4_letter_mean"] = float(np.mean([probe_acc((X_tr @ r_).numpy(), gold_of(sub_train), (Xs["fresh_conf"] @ r_).numpy(), gold_of(sets["fresh_conf"])) for r_ in rnd]))
        # coordinate predictability (ridge R2, train->confirm) and canonical correlations between carrier pairs
        pr, cc = {}, {}
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                if i == j:
                    continue
                rg = Ridge(alpha=1.0).fit(coords[j][0].numpy(), coords[i][0].numpy())
                yh = rg.predict(coords[j][1].numpy()); yt = coords[i][1].numpy()
                pr[f"{i}<-{j}"] = float(1 - ((yt - yh) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
                if i < j:
                    A, B = coords[i][0].numpy(), coords[j][0].numpy()
                    A0, B0 = A - A.mean(0), B - B.mean(0)
                    qa, qb = np.linalg.qr(A0)[0], np.linalg.qr(B0)[0]
                    cc[f"{i}-{j}"] = float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])
        ch["ridge_r2_pairs"] = pr; ch["top_canonical_corr_train"] = cc
        # retained-logit-deviation similarity between carriers (fresh confirm)
        dev_v = {k: np.array(ch["carriers"][k]["fresh_conf"]["logit_dev"]).reshape(-1, 4) for k in ch["carriers"]}
        cosd = {}
        for i in dev_v:
            for j in dev_v:
                if i < j:
                    a_, b_ = dev_v[i], dev_v[j]
                    cosd[f"{i}-{j}"] = float(np.mean((a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1) + 1e-9)))
        ch["retained_logit_dev_cosine"] = cosd
        for k in ch["carriers"]:
            ch["carriers"][k]["fresh_conf"].pop("logit_dev", None)
        # parallel fit overlap
        fp = C / f"h{h}_s{s}_k2par.pt"
        if fp.exists() and K >= 2:
            P2 = torch.load(fp); U2 = Us[1].cpu(); U1 = Us[0].cpu()
            B = torch.linalg.qr(torch.eye(4096) - U1 @ U1.T)[0]                # not used for sampling; complement sampling below
            g_ = torch.Generator().manual_seed(7)
            nulls = []
            for _ in range(100):
                V = torch.randn(4096, 4, generator=g_); V = V - U1 @ (U1.T @ V)
                nulls.append(overlap(torch.linalg.qr(V)[0], U2))
            ch["parallel_overlap_with_sequential_U2"] = dict(overlap=float(overlap(P2, U2)), top_cos=float(principal_cosines(P2, U2)[0]),
                                                             null_complement_mean=float(np.mean(nulls)), parallel_orth_to_U1=float((P2.T @ U1).abs().max()))
        # cross-chain overlaps of U1 are deferred to the report script
        H[nm] = ch
        print(nm, "K*dev", K, "orth", orth, "conf-suff", [ch["carriers"][k]["fresh_conf"]["sufficient"] for k in ch["carriers"]], flush=True)
    res["hooks"][h] = dict(acc_intact=acc_i, chains=H)
json.dump(res, open(R / "confirm_descriptive.json", "w"), indent=1, default=float)
print("S1 CONFIRM DONE", flush=True)
