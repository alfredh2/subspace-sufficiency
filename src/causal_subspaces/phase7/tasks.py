"""Phase 7 Stage 1: task generators with a common interface.

Every task yields `Item`s with per-example candidate answer tokens (`cand_ids`, class order fixed per task) and a
class index `gold`; the class is semantic (e.g. 0 = valid) so a reversed answer mapping changes only the token.
Splits are by content (entities / lexicon items / names), disjoint between train, val and test.
All generators are our own seeded re-creations (not published benchmarks) except where stated.
Raw tasks (G, N-cont) are plain text continuations without the chat template.
"""
import hashlib
import random
from dataclasses import dataclass, field

from causal_subspaces.data.logic import build_contents, build_splits
from causal_subspaces.phase05 import FORMATS as L_FORMATS, build_pairs, build_prompt as l_build_prompt

TASKS = ["L", "S", "G", "N", "M"]
FORMATS = {"L": ["num_orig", "num_rev"], "S": ["words", "num_orig", "num_rev"], "G": ["pres", "past"],
           "N": ["cont", "qa"], "M": ["letters", "digits"], "A": ["digits"]}
REVERSED = {"L": "num_rev", "S": "num_rev"}  # tasks with a reversed answer mapping


@dataclass
class Item:
    id: str
    task: str
    fmt: str
    split: str
    prompt: str
    raw: bool
    cand: list           # candidate answer strings, class order
    cand_ids: list       # their (single) token ids
    gold: int
    group: str           # independent content cluster (for clustered bootstraps)
    pair_key: str = ""
    meta: dict = field(default_factory=dict)


def _ids(tok, strings):
    out = []
    for s in strings:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{s!r} is {len(t)} tokens")
        out.append(t[0])
    return out


def _split_list(xs, rng, n_val, n_test):
    xs = list(xs)
    rng.shuffle(xs)
    return {"test": sorted(xs[:n_test]), "val": sorted(xs[n_test:n_test + n_val]), "train": sorted(xs[n_test + n_val:])}


def data_hash(items):
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it.id}|{it.prompt}|{it.gold}|{it.cand}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- L
def build_L(tok, seed=0, n_train=1200, n_pairs=160):
    sp = build_splits(seed, 256, 4, 8, False)
    ordinary = build_contents("ordinary")
    out = {}
    for fmt in FORMATS["L"]:
        f = L_FORMATS[fmt]
        cand, cids = [f["valid"], f["invalid"]], _ids(tok, [f["valid"], f["invalid"]])

        def mk(e, split):
            return Item(f"L|{fmt}|{e.id}", "L", fmt, split, l_build_prompt(fmt, e.syllogism), False, cand, cids,
                        0 if e.valid else 1, f"c{e.content_id}", meta=dict(family=e.family, form=e.form))
        rng = random.Random(seed)
        tr = sp["train"][:]
        rng.shuffle(tr)
        d = {"train": [mk(e, "train") for e in tr[:n_train]], "val": [mk(e, "val") for e in sp["val"]],
             "test": [mk(e, "test") for e in sp["items"]]}
        test_cids = sorted({e.content_id for e in sp["items"]})
        fams = sorted({e.family for e in sp["items"]})
        prs = build_pairs(ordinary, test_cids, fams, tok, fmt)
        rng.shuffle(prs)
        d["pairs"] = []
        for i, p in enumerate(prs[:n_pairs]):
            a = Item(f"L|{fmt}|p{i}v", "L", fmt, "pairs", l_build_prompt(fmt, p["valid_syl"]), False, cand, cids, 0, f"c{p['content_id']}", f"p{i}")
            b = Item(f"L|{fmt}|p{i}i", "L", fmt, "pairs", l_build_prompt(fmt, p["invalid_syl"]), False, cand, cids, 1, f"c{p['content_id']}", f"p{i}")
            d["pairs"].append((a, b))
        out[fmt] = d
    return out


# ---------------------------------------------------------------- S
POS = """wonderful excellent terrific delightful fantastic brilliant lovely charming superb impressive enjoyable pleasant
outstanding fabulous splendid marvelous amazing great good perfect beautiful gorgeous stunning magnificent fun satisfying
remarkable awesome fine nice thrilling heartwarming captivating exceptional incredible stellar enchanting inspiring""".split()
NEG = """terrible awful horrible dreadful disappointing boring dull miserable dismal poor bad unpleasant mediocre annoying
tedious frustrating lousy rotten ugly pathetic atrocious appalling disgusting depressing worthless forgettable lame nasty
weak hideous ghastly tiresome irritating abysmal shoddy dire sloppy bland""".split()
S_NOUNS = """movie film show meal hotel book concert trip game play restaurant service performance story album class product
lecture dinner party festival museum tour hike cruise match recital series season""".split()
S_TEMPL = [("The {n} was{neg} {a}.", False), ("I thought the {n} was{neg} {a}.", False), ("Honestly, the {n} seemed{neg} {a}.", False),
           ("My friend said the {n} was{neg} very {a}.", False)]
S_INSTR = {
    "words": "Is the sentiment of the following sentence positive or negative? Answer with just the word positive or negative, in lowercase, and nothing else.",
    "num_orig": "Is the sentiment of the following sentence positive or negative? Answer 1 if it is positive, or 2 if it is negative. Answer with just 1 or 2 and nothing else.",
    "num_rev": "Is the sentiment of the following sentence positive or negative? Answer 1 if it is negative, or 2 if it is positive. Answer with just 1 or 2 and nothing else.",
}
S_CAND = {"words": ["positive", "negative"], "num_orig": ["1", "2"], "num_rev": ["2", "1"]}  # class order: [positive, negative]


def s_prompt(fmt, sent):
    return f"{S_INSTR[fmt]}\n\nSentence: {sent}"


def build_S(tok, seed=0, n_train=1200, n_val=256, n_test=384, n_pairs=160):
    rng = random.Random(seed + 11)
    pos = [w for w in POS if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    neg = [w for w in NEG if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    P, N, NO = (_split_list(x, rng, 5, 8) for x in (pos, neg, S_NOUNS))
    N_ = {k: _split_list(S_NOUNS, random.Random(seed), 5, 6)[k] for k in ("train", "val", "test")}
    specs = {}
    for sp, n_items in (("train", n_train), ("val", n_val), ("test", n_test)):
        lst, seen = [], set()
        tries = 0
        while len(lst) < n_items and tries < 100000:
            tries += 1
            neg_t = rng.random() < 0.35
            t = rng.randrange(len(S_TEMPL))
            lex_pos = rng.random() < 0.5
            a = rng.choice((P if lex_pos else N)[sp])
            n = rng.choice(N_[sp])
            key = (t, neg_t, a, n)
            if key in seen:
                continue
            seen.add(key)
            sent = S_TEMPL[t][0].format(n=n, a=a, neg=" not" if neg_t else "")
            positive = lex_pos != neg_t
            lst.append(dict(sent=sent, positive=positive, adj=a, lex_pos=lex_pos, t=t, neg=neg_t, noun=n, group=f"a:{a}"))
        specs[sp] = lst
    # matched pairs on test: same template/noun/negation, polarity word swapped for an opposite-polarity word
    prs = []
    for t in range(len(S_TEMPL)):
        for neg_t in (False, True):
            for n in N_["test"]:
                a_p, a_n = rng.choice(P["test"]), rng.choice(N["test"])
                prs.append((t, neg_t, n, a_p, a_n))
    rng.shuffle(prs)
    out = {}
    for fmt in FORMATS["S"]:
        cand = S_CAND[fmt]
        cids = _ids(tok, cand)
        d = {}
        for sp, lst in specs.items():
            d[sp] = [Item(f"S|{fmt}|{sp}{i}", "S", fmt, sp, s_prompt(fmt, x["sent"]), False, cand, cids, 0 if x["positive"] else 1,
                          x["group"], meta=dict(t=x["t"], neg=x["neg"])) for i, x in enumerate(lst)]
        d["pairs"] = []
        for i, (t, neg_t, n, a_p, a_n) in enumerate(prs[:n_pairs]):
            mk = lambda a: S_TEMPL[t][0].format(n=n, a=a, neg=" not" if neg_t else "")
            ia = Item(f"S|{fmt}|p{i}a", "S", fmt, "pairs", s_prompt(fmt, mk(a_p)), False, cand, cids, 1 if neg_t else 0, f"a:{a_p}", f"p{i}")
            ib = Item(f"S|{fmt}|p{i}b", "S", fmt, "pairs", s_prompt(fmt, mk(a_n)), False, cand, cids, 0 if neg_t else 1, f"a:{a_n}", f"p{i}")
            d["pairs"].append((ia, ib))
        out[fmt] = d
    return out


# ---------------------------------------------------------------- G
G_NOUNS = """key cabinet window bottle painting letter driver teacher author farmer player picture book building garden
river bridge table chair lamp car truck bicycle student doctor nurse singer dancer artist pilot soldier sailor worker
manager clerk guard leader member reporter writer painter builder runner visitor tourist customer neighbor friend
sister brother uncle cousin daughter parent president senator officer judge lawyer engineer designer cook baker mechanic
photo poster flag banner statue fountain tower ticket package envelope contract report poem song movie""".split()
G_PREP = ["near the", "behind the", "beside the", "of the", "next to the", "across from the", "above the", "inside the"]


def _pl(w):
    return w + "s"


def build_G(tok, seed=0, n_train=1200, n_val=256, n_test=384, n_pairs=160):
    rng = random.Random(seed + 22)
    nouns = [w for w in G_NOUNS if len(tok.encode(" " + w, add_special_tokens=False)) == 1 and len(tok.encode(" " + _pl(w), add_special_tokens=False)) <= 2]
    pools = _split_list(nouns, rng, 10, 24)
    verbs = {"pres": (" is", " are", ""), "past": (" was", " were", "Yesterday, the")}
    specs = {}
    for sp, n_items in (("train", n_train), ("val", n_val), ("test", n_test)):
        lst, seen = [], set()
        while len(lst) < n_items:
            h, a = rng.sample(pools[sp], 2)
            hp, ap, pr = rng.random() < 0.5, rng.random() < 0.5, rng.choice(G_PREP)
            if (h, a, hp, ap, pr) in seen:
                continue
            seen.add((h, a, hp, ap, pr))
            lst.append((h, a, hp, ap, pr))
        specs[sp] = lst

    def text(fmt, h, a, hp, ap, pr):
        head = _pl(h) if hp else h
        att = _pl(a) if ap else a
        if fmt == "pres":
            return f"The {head} {pr} {att}"
        return f"Yesterday, the {head} {pr} {att}"
    out = {}
    for fmt in FORMATS["G"]:
        cand = list(verbs[fmt][:2])
        cids = _ids(tok, cand)
        d = {}
        for sp, lst in specs.items():
            d[sp] = [Item(f"G|{fmt}|{sp}{i}", "G", fmt, sp, text(fmt, *x), True, cand, cids, 1 if x[2] else 0, f"h:{x[0]}",
                          meta=dict(head_plural=x[2], attr_plural=x[3])) for i, x in enumerate(lst)]
        d["pairs"] = []
        prs = list(specs["test"])
        random.Random(seed + 5).shuffle(prs)
        for i, (h, a, hp, ap, pr) in enumerate(prs[:n_pairs]):
            x1, x2 = (h, a, False, ap, pr), (h, a, True, ap, pr)
            t1, t2 = text(fmt, *x1), text(fmt, *x2)
            if len(tok.encode(t1)) != len(tok.encode(t2)):
                continue
            d["pairs"].append((Item(f"G|{fmt}|p{i}s", "G", fmt, "pairs", t1, True, cand, cids, 0, f"h:{h}", f"p{i}"),
                               Item(f"G|{fmt}|p{i}p", "G", fmt, "pairs", t2, True, cand, cids, 1, f"h:{h}", f"p{i}")))
        out[fmt] = d
    return out


# ---------------------------------------------------------------- N
NAMES = """Mary John Alice Tom David Sarah James Emma Michael Laura Peter Anna Robert Julia Daniel Kate Paul Lisa Mark Helen
Steve Nancy George Linda Henry Susan Frank Carol Jack Diana Oscar Rita Hannah Nathan Maya Leo Iris Theo Victor Grace
Walter Ruth Simon Olivia Martin Clara Edward Irene Ryan Claire Adam Nora Eric Alex Jane Brian Karen Kevin Wendy Sam Tina""".split()
N_PLACE = ["store", "park", "office", "restaurant", "school", "library", "market", "beach", "station", "museum"]
N_OBJ = ["drink", "book", "letter", "gift", "ball", "note", "flower", "card"]
N_TEMPL = ["Then, {f} and {s} went to the {p}. {g} gave a {o} to",
           "When {f} and {s} arrived at the {p}, {g} handed a {o} to",
           "After {f} and {s} got to the {p}, {g} passed a {o} to",
           "{f} and {s} had a lot of fun at the {p}. {g} gave a {o} to"]
N_QA = "{pre} the other person. Who received the {o}? Answer with just the name and nothing else."


def build_N(tok, seed=0, n_train=1200, n_val=256, n_test=384, n_pairs=160):
    rng = random.Random(seed + 33)
    names = [n for n in NAMES if len(tok.encode(" " + n, add_special_tokens=False)) == 1 and len(tok.encode(n, add_special_tokens=False)) == 1]
    pools = _split_list(names, rng, 10, 18)
    specs = {}
    for sp, n_items in (("train", n_train), ("val", n_val), ("test", n_test)):
        lst, seen = [], set()
        while len(lst) < n_items:
            x, y = rng.sample(pools[sp], 2)           # x = answer (receiver), y = giver
            first_is_answer = rng.random() < 0.5
            t, p, o = rng.randrange(len(N_TEMPL)), rng.choice(N_PLACE), rng.choice(N_OBJ)
            if (x, y, first_is_answer, t, p, o) in seen:
                continue
            seen.add((x, y, first_is_answer, t, p, o))
            lst.append((x, y, first_is_answer, t, p, o))
        specs[sp] = lst

    def text(fmt, x, y, fia, t, p, o):
        f, s = (x, y) if fia else (y, x)
        base = N_TEMPL[t].format(f=f, s=s, p=p, g=y, o=o)
        if fmt == "cont":
            return base
        return N_QA.format(pre=base.rsplit(" to", 1)[0] + " to", o=o)

    def cand_of(fmt, x, y, fia):
        f, s = (x, y) if fia else (y, x)
        a = [f, s]  # class order: [first-mentioned, second-mentioned]
        return [" " + n for n in a] if fmt == "cont" else a

    out = {}
    for fmt in FORMATS["N"]:
        d = {}
        for sp, lst in specs.items():
            items = []
            for i, (x, y, fia, t, p, o) in enumerate(lst):
                cand = cand_of(fmt, x, y, fia)
                items.append(Item(f"N|{fmt}|{sp}{i}", "N", fmt, sp, text(fmt, x, y, fia, t, p, o), fmt == "cont", cand, _ids(tok, cand),
                                  0 if fia else 1, f"n:{x}", meta=dict(t=t)))
            d[sp] = items
        d["pairs"] = []
        prs = list(specs["test"])
        random.Random(seed + 6).shuffle(prs)
        for i, (x, y, fia, t, p, o) in enumerate(prs[:n_pairs]):
            a = (x, y, True, t, p, o)
            b = (x, y, False, t, p, o)
            ia, ib = (Item(f"N|{fmt}|p{i}{k}", "N", fmt, "pairs", text(fmt, *z), fmt == "cont", cand_of(fmt, *z[:3]), _ids(tok, cand_of(fmt, *z[:3])),
                           0 if z[2] else 1, f"n:{x}", f"p{i}") for k, z in (("a", a), ("b", b)))
            d["pairs"].append((ia, ib))
        out[fmt] = d
    return out


# ---------------------------------------------------------------- M
CAPITALS = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon", "Greece": "Athens",
            "Egypt": "Cairo", "Kenya": "Nairobi", "Japan": "Tokyo", "China": "Beijing", "India": "Delhi", "Russia": "Moscow",
            "Canada": "Ottawa", "Mexico": "Mexico City", "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Peru": "Lima",
            "Chile": "Santiago", "Cuba": "Havana", "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
            "Denmark": "Copenhagen", "Poland": "Warsaw", "Austria": "Vienna", "Ireland": "Dublin", "Turkey": "Ankara",
            "Iran": "Tehran", "Iraq": "Baghdad", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
            "Australia": "Canberra", "Nigeria": "Abuja", "Ghana": "Accra", "Morocco": "Rabat", "Hungary": "Budapest",
            "Romania": "Bucharest", "Ukraine": "Kyiv", "Belgium": "Brussels", "Switzerland": "Bern", "Syria": "Damascus",
            "Jordan": "Amman", "Lebanon": "Beirut", "Pakistan": "Islamabad", "Afghanistan": "Kabul", "Nepal": "Kathmandu",
            "Cambodia": "Phnom Penh", "Colombia": "Bogota", "Venezuela": "Caracas", "Ecuador": "Quito", "Bolivia": "La Paz",
            "Uruguay": "Montevideo", "Ethiopia": "Addis Ababa", "Tunisia": "Tunis", "Algeria": "Algiers", "Libya": "Tripoli",
            "Senegal": "Dakar", "Iceland": "Reykjavik"}
ELEMENTS = {"gold": "Au", "silver": "Ag", "iron": "Fe", "copper": "Cu", "lead": "Pb", "sodium": "Na", "potassium": "K",
            "calcium": "Ca", "oxygen": "O", "hydrogen": "H", "nitrogen": "N", "carbon": "C", "helium": "He", "neon": "Ne",
            "chlorine": "Cl", "sulfur": "S", "zinc": "Zn", "tin": "Sn", "mercury": "Hg", "silicon": "Si", "magnesium": "Mg",
            "aluminum": "Al", "phosphorus": "P", "nickel": "Ni", "platinum": "Pt", "uranium": "U", "iodine": "I",
            "fluorine": "F", "argon": "Ar", "lithium": "Li", "boron": "B", "titanium": "Ti", "tungsten": "W", "cobalt": "Co",
            "bromine": "Br", "radon": "Rn"}
M_Q = {"cap": "What is the capital of {e}?", "elem": "What is the chemical symbol of the element {e}?"}
M_LABELS = {"letters": ["A", "B", "C", "D"], "digits": ["1", "2", "3", "4"]}
M_END = {"letters": "Answer with just the letter A, B, C or D and nothing else.",
         "digits": "Answer with just the number 1, 2, 3 or 4 and nothing else."}


def m_prompt(fmt, q, opts):
    lab = M_LABELS[fmt]
    body = "\n".join(f"{lab[i]}. {o}" for i, o in enumerate(opts))
    return f"{q}\n{body}\n{M_END[fmt]}"


def build_M(tok, seed=0, n_train=1200):
    rng = random.Random(seed + 44)
    rel = {"cap": CAPITALS, "elem": ELEMENTS}
    ent = {r: _split_list(list(d), rng, 6 if r == "cap" else 4, 14 if r == "cap" else 8) for r, d in rel.items()}
    out = {}
    specs = {}
    for sp, reps in (("train", None), ("val", 4), ("test", 3)):
        lst = []
        qs = [(r, e) for r in rel for e in ent[r][sp]]
        if reps is None:
            while len(lst) < n_train:
                r, e = rng.choice(qs)
                lst.append((r, e, rng.randrange(4), rng.sample([x for x in ent[r][sp] if x != e], 3)))
        else:
            for r, e in qs:
                for c in range(4):
                    for _ in range(reps):
                        lst.append((r, e, c, rng.sample([x for x in ent[r][sp] if x != e], 3)))
        specs[sp] = lst
    prs = []
    for r, e in [(r, e) for r in rel for e in ent[r]["test"]]:
        for _ in range(3):
            dis = rng.sample([x for x in ent[r]["test"] if x != e], 3)
            c1, c2 = rng.sample(range(4), 2)
            prs.append((r, e, c1, c2, dis))

    def opts_of(r, e, c, dis):
        vals = [rel[r][x] for x in dis]
        ans = rel[r][e]
        return vals[:c] + [ans] + vals[c:]
    for fmt in ("letters", "digits"):
        cand = M_LABELS[fmt]
        cids = _ids(tok, cand)
        d = {}
        for sp, lst in specs.items():
            d[sp] = [Item(f"M|{fmt}|{sp}{i}", "M", fmt, sp, m_prompt(fmt, M_Q[r].format(e=e), opts_of(r, e, c, dis)), False, cand, cids, c,
                          f"e:{e}") for i, (r, e, c, dis) in enumerate(lst)]
        d["pairs"] = []
        for i, (r, e, c1, c2, dis) in enumerate(prs):
            o1 = opts_of(r, e, c1, dis)
            o2 = opts_of(r, e, c2, [x for x in dis])  # same distractors, answer moved
            # same option multiset in a different order
            d["pairs"].append((Item(f"M|{fmt}|p{i}a", "M", fmt, "pairs", m_prompt(fmt, M_Q[r].format(e=e), o1), False, cand, cids, c1, f"e:{e}", f"p{i}"),
                               Item(f"M|{fmt}|p{i}b", "M", fmt, "pairs", m_prompt(fmt, M_Q[r].format(e=e), o2), False, cand, cids, c2, f"e:{e}", f"p{i}")))
        out[fmt] = d
    return out


# ---------------------------------------------------------------- A (reserve; baseline screen only)
def build_A(tok, seed=0):
    rng = random.Random(seed + 55)
    cand = [str(i) for i in range(10)]
    cids = _ids(tok, cand)
    pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]
    rng.shuffle(pairs)
    cut = {"train": pairs[:30], "val": pairs[30:38], "test": pairs[38:]}
    d = {}
    for sp, lst in cut.items():
        d[sp] = [Item(f"A|digits|{sp}{i}", "A", "digits", sp, f"What is {a}+{b}? Answer with just the number and nothing else.", False, cand,
                      cids, a + b, f"s:{a}+{b}") for i, (a, b) in enumerate(lst)]
    return {"digits": d}


BUILDERS = {"L": build_L, "S": build_S, "G": build_G, "N": build_N, "M": build_M, "A": build_A}


def build_task(name, tok, seed=0):
    return BUILDERS[name](tok, seed)
