"""Source-position groups for position-restricted value patching (Phase 2.6, docs/phase26_plan.md).
Groups (over the eligible = non-padding source positions of a prompt):
  G1 = final token of the conclusion line, G2 = generation-prompt tail excluding the query token, Self = the query (last) token,
  Remaining = complement of G1 u G2 (INCLUDES Self), RemExSelf = Remaining minus Self, All = every eligible position."""
import zlib

import numpy as np


def spans_logic(user: str, syl_lines: list[str], family: str):
    instr_end = user.index("\n\n")
    roles = ["fact", "rule", "conclusion"] if family == "fact_first" else ["rule", "fact", "conclusion"]
    spans, pos, starts = [("instruction", 0, instr_end + 2)], instr_end + 2, []
    for ln in syl_lines:
        st = user.index(ln, pos); starts.append(st); pos = st + len(ln)
    for r, st, en in zip(roles, starts, starts[1:] + [len(user)]):
        spans.append((r, st, en))
    return spans


def categorize(tok, prompt: str, user: str, spans) -> list[str]:
    """Token category over the UNPADDED formatted prompt: first_token, chat_prefix, spans (instruction/rule/fact/conclusion), template_tail, self (last
    token); the last token of the conclusion span becomes 'conclusion_final_token'."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    u0 = prompt.index(user); u1 = u0 + len(user); cats = []
    for i, (a, b) in enumerate(off):
        if i == 0:
            c = "first_token"
        elif a < u0:
            c = "chat_prefix"
        elif a >= u1:
            c = "template_tail"
        else:
            c = "other"
            for nm, s0, s1 in spans:
                if u0 + s0 <= a < u0 + s1:
                    c = nm; break
        cats.append(c)
    cats[-1] = "self"
    last_stmt = max((i for i, c in enumerate(cats) if c == "conclusion" and i < len(cats) - 1), default=None)
    if last_stmt is not None:
        cats[last_stmt] = "conclusion_final_token"
    return cats


def group_masks(cats: list[str]) -> dict:
    c = np.array(cats)
    n = len(c)
    g1, g2, selfm = c == "conclusion_final_token", c == "template_tail", c == "self"
    allm = np.ones(n, bool)
    rem = ~(g1 | g2)
    return dict(G1=g1, G2=g2, G1G2=g1 | g2, Self=selfm, Rem=rem, RemExSelf=rem & ~selfm, All=allm)


def check_groups(m: dict) -> list[str]:
    """Return a list of violated assertions (empty = fine)."""
    bad = []
    if (m["G1"] & m["G2"]).any() or (m["G1"] & m["Self"]).any() or (m["G2"] & m["Self"]).any():
        bad.append("G1/G2/Self not pairwise disjoint")
    if (m["G1G2"] & m["Rem"]).any() or not (m["G1G2"] | m["Rem"]).all():
        bad.append("G1uG2 and Remaining are not a partition of the eligible positions")
    if (m["Self"] & ~m["Rem"]).any():
        bad.append("Self not inside Remaining")
    if (m["RemExSelf"] & m["Self"]).any() or ((m["RemExSelf"] | m["Self"]) != m["Rem"]).any():
        bad.append("RemExSelf inconsistent")
    if m["G1"].sum() != 1 or m["Self"].sum() != 1 or m["G2"].sum() < 1:
        bad.append("unexpected group sizes")
    return bad


def pad_mask(mask: np.ndarray, T: int) -> np.ndarray:
    """Left-pad an unpadded-sequence mask to the padded length T (padding positions are never selected)."""
    out = np.zeros(T, bool)
    out[T - len(mask):] = mask
    return out


def random_positions(pool: np.ndarray, count: int, tag: str, trial: int, draw: int):
    """Deterministic draw of `count` distinct positions from `pool` (eligible, non-padding, causal). If the pool is too small all pool positions are used
    and the draw is flagged. Seeds come from crc32 of a fixed string (never Python's randomised hash)."""
    rng = np.random.default_rng(zlib.crc32(f"{tag}|{trial}|{draw}".encode()))
    if len(pool) < count:
        return np.sort(pool), True
    return np.sort(rng.choice(pool, count, replace=False)), False
