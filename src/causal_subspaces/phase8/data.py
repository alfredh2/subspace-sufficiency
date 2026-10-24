"""Phase 8 fresh QA entities (never in any earlier lexicon or split). Entity split assignment is made by a seeded shuffle BEFORE the
intact-model baseline screen; entities failing the screen (intact accuracy < 0.9) are dropped before any intervention."""
import hashlib, json, random

from causal_subspaces.phase7.tasks import Item, M_LABELS, M_Q, CAPITALS, ELEMENTS, m_prompt, _ids

FRESH_CAPITALS = {
    "Albania": "Tirana", "Armenia": "Yerevan", "Azerbaijan": "Baku", "Bangladesh": "Dhaka", "Belarus": "Minsk", "Bulgaria": "Sofia",
    "Croatia": "Zagreb", "Estonia": "Tallinn", "Jamaica": "Kingston", "Latvia": "Riga", "Lithuania": "Vilnius", "Malta": "Valletta",
    "Mongolia": "Ulaanbaatar", "Mozambique": "Maputo", "Namibia": "Windhoek", "New Zealand": "Wellington", "Panama": "Panama City",
    "Paraguay": "Asuncion", "Philippines": "Manila", "Serbia": "Belgrade", "Slovakia": "Bratislava", "Slovenia": "Ljubljana",
    "Sudan": "Khartoum", "Uganda": "Kampala", "Zambia": "Lusaka", "Zimbabwe": "Harare", "Angola": "Luanda", "Cameroon": "Yaounde",
    "Madagascar": "Antananarivo", "Rwanda": "Kigali", "Laos": "Vientiane", "Qatar": "Doha", "Oman": "Muscat", "Saudi Arabia": "Riyadh",
    "Cyprus": "Nicosia", "Honduras": "Tegucigalpa", "Nicaragua": "Managua", "Bhutan": "Thimphu", "Botswana": "Gaborone", "Mali": "Bamako",
    "Liberia": "Monrovia", "Uzbekistan": "Tashkent", "Czechia": "Prague", "Costa Rica": "San Jose"}
FRESH_ELEMENTS = {
    "antimony": "Sb", "arsenic": "As", "barium": "Ba", "beryllium": "Be", "bismuth": "Bi", "cadmium": "Cd", "chromium": "Cr", "gallium": "Ga",
    "iridium": "Ir", "krypton": "Kr", "manganese": "Mn", "molybdenum": "Mo", "palladium": "Pd", "plutonium": "Pu", "radium": "Ra",
    "selenium": "Se", "strontium": "Sr", "vanadium": "V", "xenon": "Xe", "zirconium": "Zr", "cesium": "Cs", "germanium": "Ge",
    "tellurium": "Te", "rubidium": "Rb", "thorium": "Th"}
ASSIGN_SEED = 8001
N_DEV = {"cap": 12, "elem": 7}          # remaining go to confirm (32 + 18 before the screen)
REL = {"cap": FRESH_CAPITALS, "elem": FRESH_ELEMENTS}


def check_disjoint():
    assert not (set(FRESH_CAPITALS) & set(CAPITALS)) and not (set(FRESH_ELEMENTS) & set(ELEMENTS))
    assert not (set(FRESH_CAPITALS.values()) & set(CAPITALS.values())) and not (set(FRESH_ELEMENTS.values()) & set(ELEMENTS.values()))
    assert len(FRESH_CAPITALS) == 44 and len(FRESH_ELEMENTS) == 25


def assign():
    """Seeded entity-to-split assignment, made before any model run."""
    rng = random.Random(ASSIGN_SEED)
    out = {}
    for r, d in REL.items():
        xs = sorted(d)
        rng.shuffle(xs)
        out[r] = {"devf": sorted(xs[:N_DEV[r]]), "conf": sorted(xs[N_DEV[r]:])}
    return out


def build_items(tok, ents, fmt="letters", reps=2, seed=0, splits=("devf", "conf")):
    """ents: {rel: {split: [entities]}}. Four answer positions x `reps` distractor draws per entity; 3 length-matched transplant pairs per
    entity (same distractors, answer moved). Distractors come from the same split's entity list."""
    rng = random.Random(seed + 66)
    cand = M_LABELS[fmt]
    cids = _ids(tok, cand)
    out = {}
    for sp in splits:
        def opts_of(r, e, c, dis):
            vals = [REL[r][x] for x in dis]
            return vals[:c] + [REL[r][e]] + vals[c:]
        items, pairs = [], []
        for r in REL:
            for e in ents[r][sp]:
                pool = [x for x in ents[r][sp] if x != e]
                for c in range(4):
                    for _ in range(reps):
                        dis = rng.sample(pool, 3)
                        items.append(Item(f"M8|{fmt}|{sp}{len(items)}", "M", fmt, sp, m_prompt(fmt, M_Q[r].format(e=e), opts_of(r, e, c, dis)), False,
                                          cand, cids, c, f"e:{e}", meta={"rel": r}))
                for _ in range(3):
                    dis = rng.sample(pool, 3)
                    c1, c2 = rng.sample(range(4), 2)
                    k = len(pairs)
                    pairs.append((Item(f"M8|{fmt}|{sp}p{k}a", "M", fmt, sp, m_prompt(fmt, M_Q[r].format(e=e), opts_of(r, e, c1, dis)), False, cand, cids,
                                       c1, f"e:{e}", f"p{k}", meta={"rel": r}),
                                  Item(f"M8|{fmt}|{sp}p{k}b", "M", fmt, sp, m_prompt(fmt, M_Q[r].format(e=e), opts_of(r, e, c2, dis)), False, cand, cids,
                                       c2, f"e:{e}", f"p{k}", meta={"rel": r})))
        out[sp] = dict(items=items, pairs=pairs)
    return out


def items_hash(items):
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it.id}|{it.prompt}|{it.gold}".encode())
    return h.hexdigest()[:16]
