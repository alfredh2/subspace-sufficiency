"""MVS optimisation (paper Sec. 3 Eq. 2, Sec. 4, App. C).

Faithful: Adam on V (d x k); U = QR(V) at every step; loss = KL(p_original || p_retained) over the full
vocabulary at the final prompt position, averaged over prompts; 1600 steps, 16 prompts/step, lr 1e-2 cosine
annealed to lr/20; on-manifold init from the activation covariance; projection at every token position.
Approximations / unstated (configurable): gradient through QR (we differentiate through it), exact cosine
formula, sampling prompts i.i.d. per step from the optimisation pool, no weight decay / clipping.
"""
import math
import time

import numpy as np
import torch

from causal_subspaces.evaluation.scoring import evaluate_prompts
from causal_subspaces.models.hooks import residual_intervention
from causal_subspaces.models.loading import encode, format_prompt, last_logits
from causal_subspaces.mvs.checkpoint import latest_checkpoint, save_checkpoint, load_checkpoint
from causal_subspaces.mvs.projection import orthonormality_error, orthonormalize


def kl_teacher_student(teacher_logp: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Per-example KL(p_teacher || p_student), fp32, full vocabulary. Terms with p=0 contribute 0 (exp(-inf))."""
    logq = torch.log_softmax(student_logits.float(), -1)
    p = teacher_logp.exp()
    return (p * (teacher_logp - logq)).sum(-1)


@torch.no_grad()
def teacher_logprobs(model, tok, examples, cfg_enable_thinking, device, batch_size) -> torch.Tensor:
    """Log-probs of the UNMODIFIED model at the final prompt position, (N, V) fp32 on CPU."""
    prompts = [format_prompt(tok, e.prompt_text, cfg_enable_thinking) for e in examples]
    out = torch.empty(len(prompts), model.config.vocab_size, dtype=torch.float32)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, pos = encode(tok, [prompts[i] for i in idx], device)
        out[idx] = torch.log_softmax(last_logits(model, ids, mask, pos), -1).cpu()
    return out


def lr_at(step: int, total: int, lr: float, final_div: float = 20.0) -> float:
    lo = lr / final_div
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * step / total))


def train_mvs(model, tok, splits, mu, V0, cfg, run, device="cuda", teacher=None, resume=False):
    t, hw = cfg["train"], cfg["hardware"]
    idx_res, k = cfg["mvs"]["residual_index"], cfg["mvs"]["rank"]
    think = cfg["model"]["enable_thinking"]
    train_ex, val_ex = splits["train"], splits["val"]
    mu = mu.to(device)
    V = torch.nn.Parameter(V0.clone().to(device))
    opt = torch.optim.Adam([V], lr=t["lr"], betas=tuple(t.get("betas", (0.9, 0.999))), weight_decay=t.get("weight_decay", 0.0))
    rng = np.random.default_rng(cfg["seed"])
    step0 = 0
    ckdir = run.path / "checkpoints"
    if resume and (ck := latest_checkpoint(ckdir)):
        c = load_checkpoint(ck)
        V.data.copy_(c["V"].to(device)); opt.load_state_dict(c["optimizer"]); step0 = c["step"]
        rng.bit_generator.state = c["rng"]
    if teacher is None:
        teacher = teacher_logprobs(model, tok, train_ex, think, device, hw["eval_batch_size"])
    val_teacher = teacher_logprobs(model, tok, val_ex, think, device, hw["eval_batch_size"])
    prompts = [format_prompt(tok, e.prompt_text, think) for e in train_ex]
    B, mb = t["batch_size"], hw["micro_batch_size"]
    assert B % mb == 0
    ckpt = cfg["model"].get("gradient_checkpointing", False)
    model.train(ckpt)  # HF only applies activation checkpointing in train mode; Qwen3 has zero dropout
    log = run.jsonl("training_log.jsonl", "a" if step0 else "w")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    kls = []
    for step in range(step0, t["steps"]):
        lr = lr_at(step, t["steps"], t["lr"], t.get("final_lr_div", 20.0))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rng.choice(len(train_ex), B, replace=False)
        opt.zero_grad(set_to_none=True)
        U = orthonormalize(V)
        Ud = U.detach().requires_grad_(True)  # leaf: accumulate micro-batch grads, backprop through QR once
        step_kl = 0.0
        for s in range(0, B, mb):
            bi = batch[s:s + mb]
            ids, mask, pos = encode(tok, [prompts[i] for i in bi], device)
            with residual_intervention(model, idx_res, "retain", Ud, mu):
                logits = last_logits(model, ids, mask, pos)
            kl = kl_teacher_student(teacher[bi].to(device), logits)
            (kl.sum() / B).backward()
            step_kl += kl.sum().item() / B
        U.backward(Ud.grad)
        gn = V.grad.norm().item()
        if t.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_([V], t["grad_clip"])
        opt.step()
        kls.append(step_kl)
        rec = dict(step=step + 1, kl=step_kl, grad_norm=gn, lr=lr, orth_err=orthonormality_error(orthonormalize(V.detach())),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        if (step + 1) % t["eval_every"] == 0 or step + 1 == t["steps"]:
            rec.update(evaluate_val(model, tok, val_ex, val_teacher, orthonormalize(V.detach()), mu, cfg, device, ckpt))
        log.write(**rec)
        if (step + 1) % t["ckpt_every"] == 0 or step + 1 == t["steps"]:
            save_checkpoint(ckdir / f"step_{step + 1}.pt", V=V, U=orthonormalize(V.detach()), mu=mu, step=step + 1,
                            optimizer=opt, rng=rng.bit_generator.state,
                            meta=dict(residual_index=idx_res, rank=k, seed=cfg["seed"]))
    log.close()
    model.eval()
    U = orthonormalize(V.detach())
    return U, dict(final_kl_median_last5=float(np.median(kls[-5:])), steps=t["steps"],
                   train_seconds=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)


@torch.no_grad()
def evaluate_val(model, tok, val_ex, val_teacher, U, mu, cfg, device, was_train):
    """Validation KL (retained vs original) and primary retention/ablation accuracy on the VAL split only."""
    model.eval()
    hw = cfg["hardware"]
    idx = cfg["mvs"]["residual_index"]
    prompts = [format_prompt(tok, e.prompt_text, cfg["model"]["enable_thinking"]) for e in val_ex]
    kls = []
    for s in range(0, len(prompts), hw["eval_batch_size"]):
        ids, mask, pos = encode(tok, prompts[s:s + hw["eval_batch_size"]], device)
        with residual_intervention(model, idx, "retain", U, mu):
            lg = last_logits(model, ids, mask, pos)
        kls.append(kl_teacher_student(val_teacher[s:s + len(ids)].to(device), lg).cpu())
    gold = [e.label for e in val_ex]
    texts = [e.prompt_text for e in val_ex]
    kw = dict(device=device, batch_size=hw["eval_batch_size"], residual_index=idx, U=U, mu=mu)
    r = evaluate_prompts(model, tok, texts, gold, ["1", "2"], mode="retain", **kw)
    a = evaluate_prompts(model, tok, texts, gold, ["1", "2"], mode="ablate", **kw)
    model.train(was_train)
    return dict(val_kl=float(torch.cat(kls).mean()), val_retain_acc=float(np.mean([x["correct"] for x in r])),
                val_retain_invalid=float(np.mean([x["invalid"] for x in r])),
                val_ablate_acc=float(np.mean([x["correct"] for x in a])),
                val_ablate_invalid=float(np.mean([x["invalid"] for x in a])))
