"""Re-created conditional-inference benchmark following Kean & Boix-Adsera, Sec. 4.1 / App. B.1.

THIS IS NOT THE PAPER'S DATASET (none was released). What is taken from the paper vs. invented:
  faithful   : instruction text; answer labels (1 = valid, 2 = invalid); 4 argument forms x {affirmative,
               negated} consequent = 8 forms; 14 template-family names; the *rule* wording of every
               family (Table 2); the negation wording rules; 6 conclusion markers; held-out families
               (suppose_observe, whenever, subset); 2 lexicons (ordinary / nonword) x 48 contents; the
               three 256-item evaluation splits (Items / Families / Nonwords).
  approximate: fact and conclusion sentences (paper gives only one worked example); the 48 contents and
               nonword lexicon; premise order of non-`fact_first` families (rule first, as stated);
               optimisation/validation/test partition of contents; the 'which term the rule mentions
               first' surface factor is not implemented; the 32 stem-sharing event items are avoided by
               construction instead of removed, so we have 10,752 items rather than 10,720.
Each example carries `fidelity` metadata so reports can separate the two.
"""
import hashlib
import random
from dataclasses import asdict, dataclass

from causal_subspaces.utils.reproducibility import stable_hash

INSTRUCTION = (  # verbatim from Sec. 4.1
    "Please tell me whether the following reasoning is valid or invalid (i.e., does the conclusion "
    "necessarily follow from the premises?). Answer 1 if it is valid, or 2 if it is invalid. "
    "Answer with just 1 or 2 and nothing else."
)
MARKERS = ["Therefore,", "Hence,", "Thus,", "So,", "It follows that", "Conclusion:"]

# form -> (fact cites antecedent?, fact affirms?, conclusion cites antecedent?, conclusion affirms?, valid)
# For rule A => Q where Q is B (affirmative consequent) or not-B (negated consequent); "affirm" is relative to
# the rule's own literal: affirming Q means asserting the consequent literal of the rule.
FORMS = {
    "MP": dict(fact="A", fact_pos=True, concl="Q", concl_pos=True, valid=True),
    "DA": dict(fact="A", fact_pos=False, concl="Q", concl_pos=False, valid=False),
    "AC": dict(fact="Q", fact_pos=True, concl="A", concl_pos=True, valid=False),
    "MT": dict(fact="Q", fact_pos=False, concl="A", concl_pos=False, valid=True),
}
TRAIN_FAMILIES = ["if_then", "fact_first", "reversed_if", "reversed_whenever", "always_generic", "all_are",
                  "all_nouns", "if_then_event", "every_time_event", "guarantee", "semi_formal"]
HELDOUT_FAMILIES = ["suppose_observe", "whenever", "subset"]
FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES

NOUNS = """cube desk lamp shelf coin rope chair bottle mirror basket ladder candle kettle pillow button bucket
blanket window ribbon hammer jacket pencil wagon drum bridge boat door clock statue towel spoon plate sofa tent
flag wheel key brick barrel fence gate sack tray vase helmet cushion ball stool""".split()
ADJ_A = """little tall heavy round soft old cool wide narrow smooth shiny rough warm thick thin light dull hard flat
bright quiet loud rusty dusty new long short deep sharp dry wet plain fancy empty full hollow solid tidy cheap
fragile stiff bent steep crooked grand tiny big mild""".split()
ADJ_B = """yellow pink green red blue purple orange brown gray white black golden silver striped spotted clean dirty
valuable famous popular useful stylish sturdy ancient modern lovely ugly curious gentle fierce lucky magical
ordinary precious sacred hidden stolen borrowed polished painted wooden metallic plastic woolen glassy leafy
mossy sandy""".split()
CAT_NOUNS = """bolt wire tile rod plank stone coil pipe disk ring chain panel rail pin nail screw beam block cable
spring hook lever gear valve lens mask badge ticket token card stamp scroll tablet shell feather pebble bead
thread cork wedge crate jar cup bowl pot pan can box""".split()
VERBS = """whistle hum rattle creak squeak glow flicker shake spin rumble buzz chirp tick chime wobble tremble
vibrate hiss crackle rustle clatter jingle sparkle shimmer drip bounce roll slide sway swing float drift tumble
flutter pulse throb blink clink clank thump snap pop whir murmur whisper echo groan sing""".split()


def plural(w: str) -> str:
    if w.endswith("f"):
        return w[:-1] + "ves"
    return w + ("es" if w.endswith(("s", "x", "sh", "ch", "z")) else "s")


def third(v: str) -> str:
    return v + ("es" if v.endswith(("s", "x", "sh", "ch", "z")) else "s")


def _nonwords(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    onsets = ["bl", "dr", "fl", "gr", "kl", "pr", "st", "sk", "tr", "zw", "br", "cl", "gl", "sp", "b", "d", "f",
              "g", "k", "l", "m", "n", "p", "r", "t", "v", "w", "z", "sh", "ch", "th"]
    vowels = ["a", "e", "i", "o", "u", "oo", "ee", "ai"]
    codas = ["", "m", "n", "p", "t", "k", "b", "d", "g", "l", "r", "nd", "mp"]
    out, seen = [], set()
    while len(out) < n:
        w = "".join(rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas) for _ in range(rng.choice([1, 2])))
        if 4 <= len(w) <= 8 and w not in seen and not w.endswith(("s", "x", "z")):
            seen.add(w)
            out.append(w)
    return out


def build_contents(lexicon: str, n: int = 48, seed: int = 0) -> list[dict]:
    """Each content: noun (entity), adj_a (antecedent), adj_b (consequent), cat (category noun),
    ev1/ev2 = (noun, base verb) for the event families."""
    if lexicon == "ordinary":
        assert all(len(x) == n for x in (NOUNS, ADJ_A, ADJ_B, CAT_NOUNS, VERBS))
        return [dict(noun=NOUNS[i], adj_a=ADJ_A[i], adj_b=ADJ_B[i], cat=CAT_NOUNS[i],
                     ev1=(NOUNS[i], VERBS[i]), ev2=(NOUNS[(i + 7) % n], VERBS[(i + 24) % n])) for i in range(n)]
    w = _nonwords(8 * n, seed + 1234)
    return [dict(noun=w[8 * i], adj_a=w[8 * i + 1], adj_b=w[8 * i + 2], cat=w[8 * i + 3],
                 ev1=(w[8 * i + 4], w[8 * i + 5]), ev2=(w[8 * i + 6], w[8 * i + 7])) for i in range(n)]


# ---- sentence realisation -------------------------------------------------------------------------
def _lit(c: dict, fam: str, which: str, pos: bool) -> str:
    """Sentence about a particular entity asserting (pos) / denying (not pos) the antecedent A or consequent B."""
    n = c["noun"]
    if fam in ("if_then_event", "every_time_event"):
        noun, verb = c["ev1"] if which == "A" else c["ev2"]
        return f"The {noun} {third(verb)}." if pos else f"The {noun} does not {verb}."
    if fam == "semi_formal":
        p = "P" if which == "A" else "W"
        return f"{p} is true." if pos else f"{p} is false."
    if fam in ("all_are", "all_nouns"):
        if which == "A":
            return f"The object is a {n}." if pos else f"The object is not a {n}."
        if fam == "all_are":
            return f"The object is {c['adj_b']}." if pos else f"The object is not {c['adj_b']}."
        return f"The object is a {c['cat']}." if pos else f"The object is not a {c['cat']}."
    if fam == "subset":
        s = n if which == "A" else c["cat"]
        return (f"The object belongs to the set of {plural(s)}." if pos
                else f"The object does not belong to the set of {plural(s)}.")
    ent = f"This {n}" if fam == "always_generic" else f"The {n}"
    adj = c["adj_a"] if which == "A" else c["adj_b"]
    return f"{ent} is {adj}." if pos else f"{ent} is not {adj}."


def _rule(c: dict, fam: str, neg_q: bool) -> str:
    n, a, b = c["noun"], c["adj_a"], c["adj_b"]
    nb = f"not {b}"
    bq = nb if neg_q else b
    if fam in ("if_then", "fact_first"):
        return f"If the {n} is {a}, then it is {bq}."
    if fam == "reversed_if":
        return f"The {n} is {bq} if it is {a}."
    if fam == "suppose_observe":
        return f"Suppose that if the {n} is {a}, then it is {bq}."
    if fam == "whenever":
        return f"Whenever the {n} is {a}, it is {bq}."
    if fam == "reversed_whenever":
        return f"The {n} is {bq} whenever it is {a}."
    if fam == "always_generic":
        return f"{plural(n).capitalize()} that are {a} are {'never' if neg_q else 'always'} {b}."
    if fam == "all_are":
        return f"{'No' if neg_q else 'All'} {plural(n)} are {b}."
    if fam == "all_nouns":
        return f"{'No' if neg_q else 'All'} {plural(n)} are {plural(c['cat'])}."
    if fam == "subset":
        rel = "is disjoint from" if neg_q else "is a subset of"
        return f"The set of {plural(n)} {rel} the set of {plural(c['cat'])}."
    if fam in ("if_then_event", "every_time_event"):
        (n1, v1), (n2, v2) = c["ev1"], c["ev2"]
        cons = f"the {n2} does not {v2}" if neg_q else f"the {n2} {third(v2)}"
        return (f"If the {n1} {third(v1)}, then {cons}." if fam == "if_then_event"
                else f"Every time the {n1} {third(v1)}, {cons}.")
    if fam == "guarantee":
        return f"Being {a} guarantees that the {n} is {bq}."
    if fam == "semi_formal":
        return f"Rule: if P then {'not ' if neg_q else ''}W."
    raise KeyError(fam)


def _concl(sentence: str, marker: str) -> str:
    if sentence.startswith(("The ", "This ")):
        sentence = sentence[0].lower() + sentence[1:]
    return f"{marker} {sentence}"


def build_syllogism(c: dict, fam: str, form: str, neg_q: bool, marker: str) -> str:
    f = FORMS[form]
    # Literal polarity relative to the rule: when the consequent is negated (Q = not B), "affirming Q"
    # means asserting not-B, so the B-literal polarity flips; A is never negated in the rule.
    def sent(which_lit: str, pos: bool) -> str:
        if which_lit == "A":
            return _lit(c, fam, "A", pos)
        return _lit(c, fam, "B", pos != neg_q)  # Q = B (pos stays) or not-B (pos flips)
    fact = sent(f["fact"], f["fact_pos"])
    concl = _concl(sent(f["concl"], f["concl_pos"]), marker)
    rule = _rule(c, fam, neg_q)
    lines = [fact, rule, concl] if fam == "fact_first" else [rule, fact, concl]
    return "\n".join(lines)


@dataclass
class Example:
    id: str
    split: str
    family: str
    form: str
    valid: bool
    label: str  # "1" valid / "2" invalid
    neg_consequent: bool
    lexicon: str
    content_id: int
    marker: str
    syllogism: str
    prompt_text: str  # user-message text (instruction + syllogism)
    fidelity: dict

    def to_dict(self):
        return asdict(self)


def _fidelity(fam: str) -> dict:
    return {"rule_wording": "faithful (Table 2)", "fact_and_conclusion": "approximation",
            "contents": "approximation", "instruction": "faithful (verbatim)"}


def make_example(lexicon, content_id, content, fam, form, neg_q, split) -> Example:
    key = f"{lexicon}|{content_id}|{fam}|{form}|{neg_q}"
    marker = MARKERS[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(MARKERS)]
    syl = build_syllogism(content, fam, form, neg_q, marker)
    valid = FORMS[form]["valid"]
    return Example(id=hashlib.md5(key.encode()).hexdigest()[:10], split=split, family=fam,
                   form=form + ("_negQ" if neg_q else ""), valid=valid, label="1" if valid else "2",
                   neg_consequent=neg_q, lexicon=lexicon, content_id=content_id, marker=marker,
                   syllogism=syl, prompt_text=f"{INSTRUCTION}\n\n{syl}", fidelity=_fidelity(fam))


def _all(lexicon, contents, content_ids, families, split) -> list[Example]:
    return [make_example(lexicon, i, contents[i], fam, form, neg, split)
            for i in content_ids for fam in families for form in FORMS for neg in (False, True)]


def _stratified_sample(exs: list[Example], n: int, rng: random.Random) -> list[Example]:
    """Round-robin over (family, form) strata so each is as evenly represented as possible."""
    strata: dict = {}
    for e in exs:
        strata.setdefault((e.family, e.form), []).append(e)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    out, i = [], 0
    while len(out) < min(n, len(exs)):
        k = keys[i % len(keys)]
        if strata[k]:
            out.append(strata[k].pop())
        i += 1
    return out


def build_splits(seed: int = 0, n_eval: int = 256, n_val_contents: int = 4, n_test_contents: int = 8,
                 nonwords_in_optimization: bool = False) -> dict[str, list[Example]]:
    """Splits (all deterministic in `seed`):
      train      : ordinary lexicon, train contents, 11 train families           (MVS optimisation pool)
      val        : ordinary lexicon, val contents,   11 train families           (KL monitoring / layer-convention sweep)
      items      : ordinary lexicon, test contents,  11 train families  [256]    (paper 'Items')
      families   : ordinary lexicon, all contents,   3 held-out families [256]   (paper 'Families')
      nonwords   : nonword lexicon,  all contents,   11 train families  [256]    (paper 'Nonwords')
    Content ids are disjoint between train / val / items, so no lexical-combination leakage.
    """
    rng = random.Random(seed)
    ordinary, nonword = build_contents("ordinary"), build_contents("nonword", seed=seed)
    ids = list(range(len(ordinary)))
    rng.shuffle(ids)
    test_ids, val_ids = sorted(ids[:n_test_contents]), sorted(ids[n_test_contents:n_test_contents + n_val_contents])
    train_ids = sorted(ids[n_test_contents + n_val_contents:])
    s = {
        "train": _all("ordinary", ordinary, train_ids, TRAIN_FAMILIES, "train"),
        "val": _all("ordinary", ordinary, val_ids, TRAIN_FAMILIES, "val"),
        "items": _stratified_sample(_all("ordinary", ordinary, test_ids, TRAIN_FAMILIES, "items"), n_eval, rng),
        "families": _stratified_sample(_all("ordinary", ordinary, ids, HELDOUT_FAMILIES, "families"), n_eval, rng),
        "nonwords": _stratified_sample(_all("nonword", nonword, ids, TRAIN_FAMILIES, "nonwords"), n_eval, rng),
    }
    if nonwords_in_optimization:  # off by default: paper does not say; nonwords are meant as a confound-free probe
        s["train"] += _all("nonword", nonword, train_ids, TRAIN_FAMILIES, "train")
    return s


def dataset_hash(splits: dict[str, list[Example]]) -> str:
    return stable_hash({k: [e.id + e.syllogism for e in v] for k, v in splits.items()})
