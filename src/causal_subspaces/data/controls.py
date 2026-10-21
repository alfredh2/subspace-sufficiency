"""Non-logic control tasks. Prompt TEXT is verbatim from the paper (App. B.2/B.3); item generation (word lists,
distractors, sampling, counts) is NOT specified there, so every item set here is our seeded re-creation
(`fidelity` = "prompt faithful, items approximate"). Raven is skipped (paper: uninterpretable for Qwen, 0.54
intact). Scoring: first generated character vs. label (same as logic); multiplication uses the leading number
of an 8-token greedy generation.
"""
import random

FORCED = "Answer with just 1 or 2 and nothing else."
WORDS = """dolphin tiger urchin mitten hornet kettle barrel summit pencil anchor ribbon lantern walnut compass
harbor pillow saddle thimble copper meadow ladder basket falcon candle marble orchard puppet quartz rocket
sponge tunnel violin whistle yogurt zipper blanket cabinet donkey engine feather garden hammer island jacket
kitchen lemon mirror napkin ostrich parrot rabbit sailor teapot umbrella village window wagon""".split()
DISTRACTORS = "velvet silence justice courage melody rumor freedom dream thunder shadow".split()


def word_recall(n, list_len, rng):
    items = []
    for _ in range(n):
        w = rng.sample(WORDS, list_len)
        pos, neg = rng.choice(w), rng.choice(DISTRACTORS)
        opts = [pos, neg]
        flip = rng.random() < 0.5
        if flip:
            opts.reverse()
        text = (f"Here is a list of words:\n{', '.join(w)}\n\nWhich of these two words appeared in the list above? "
                f"{FORCED}\n1: {opts[0]}\n2: {opts[1]}")
        items.append((text, "2" if flip else "1"))
    return items


SENTS = ["The old man walked slowly along the river bank.", "She opened the window to let in some fresh air.",
         "Several students gathered outside the library after class.", "The train arrived late because of the heavy snow.",
         "He poured a cup of coffee and read the newspaper.", "A small dog chased the ball across the green field.",
         "They decided to build a wooden fence around the garden.", "The committee will announce its decision next week."]


def _pseudo(s, rng):
    out = []
    for w in s.rstrip(".").split():
        cut = max(2, len(w) - rng.choice([0, 1, 2]))
        t = list(w[:cut])
        for i in range(1, len(t)):
            if t[i] in "aeiou" and rng.random() < .6:
                t[i] = rng.choice("aeiou")
        out.append("".join(t) + rng.choice(["", "s", "en", "ed"]))
    return " ".join(out).capitalize() + "."


def localizer(n, rng):
    head = ("Please read the following text. Is it an English sentence, or a nonsensical set of words? Answer 1 if it is "
            "an English sentence, or 2 if it is a nonsensical set of words. Answer with just 1 or 2 and nothing else.")
    out = []
    for _ in range(n):
        s = rng.choice(SENTS)
        if rng.random() < .5:
            out.append((f'{head}\n\n"{s}"', "1"))
        else:
            out.append((f'{head}\n\n"{_pseudo(s, rng)}"', "2"))
    return out


def prontoqa(n, rng):
    """1-3 hop positives; negatives are converse errors or unrelated stated branches (paper's description)."""
    syl = ["cump", "plint", "grix", "defot", "corple", "snick", "blint", "tromp", "trant", "lemp", "wumpus", "rompus",
           "dalpus", "yumpus", "zumpus", "numpus", "gorpus", "jompus", "vumpus", "impus"]
    names = ["Hannah", "Nate", "Rita", "Oscar", "Maya", "Leo", "Iris", "Theo"]
    out = []
    for _ in range(n):
        c = rng.sample(syl, 8)
        hops = rng.choice([1, 2, 3])
        chain = c[:hops + 1]  # chain[0] <- ... : person is chain[0]; "All chain[i] are chain[i+1]"
        facts = [f"All {chain[i]}s are {chain[i + 1]}s." for i in range(hops)]
        distract = [f"All {c[5]}s are {c[6]}s.", f"All {c[7]}s are {chain[min(1, hops)]}s."]
        person = rng.choice(names)
        pos = rng.random() < .5
        if pos:
            q, gold = chain[-1], "1"
        elif rng.random() < .5:  # converse error: ask about a category that is only a *source* of the chain
            q, gold = c[7], "2"   # c[7] -> chain[1]; person is chain[0] -> not derivable
        else:
            q, gold = c[6], "2"   # unrelated stated branch (mentioned in premises)
        sents = facts + distract
        rng.shuffle(sents)
        text = (f"Answer 1 if the statement is true, or 2 if it is false. {FORCED}\n{' '.join(sents)} "
                f"{person} is a {chain[0]}.\nIs {person} a {q}?")
        out.append((text, gold))
    return out


def mmlu(n_per_subject, subjects, rng):
    from datasets import load_dataset
    out = []
    for sub in subjects:
        ds = load_dataset("cais/mmlu", sub, split="test")
        for r in rng.sample(list(ds), min(n_per_subject, len(ds))):
            gold = r["choices"][r["answer"]]
            dist = rng.choice([c for i, c in enumerate(r["choices"]) if i != r["answer"]])
            opts = [gold, dist]
            flip = rng.random() < .5
            if flip:
                opts.reverse()
            text = ("Answer the following question. Reply with EXACTLY ONE CHARACTER -- 1 or 2 -- and nothing else. Do not "
                    f'write "Answer", punctuation, or any explanation.\n{r["question"]}\n1. {opts[0]}\n2. {opts[1]}\nAnswer:')
            out.append((text, "2" if flip else "1"))
    return out


def stroop(n, rng):
    cols = ["red", "green", "blue", "yellow"]
    out = []
    for _ in range(n):
        word, ink = rng.choice(cols), rng.choice(cols)
        other = rng.choice([c for c in cols if c != ink])
        opts = [ink, other]
        flip = rng.random() < .5
        if flip:
            opts.reverse()
        out.append((f"You will be shown a word and the ink color it is printed in. Respond with the ink color, not the word "
                    f"itself.\n\nThe word '{word.upper()}' is printed in {ink} ink.\n\n1. {opts[0]}\n2. {opts[1]}\n\n"
                    "Reply with EXACTLY ONE CHARACTER -- 1 or 2 -- and nothing else.", "2" if flip else "1",
                    "congruent" if word == ink else "incongruent"))
    return out


def msit(n, rng):
    out = []
    for _ in range(n):
        a, b = rng.sample(range(1, 4), 2)
        digs = [a, a, b]
        rng.shuffle(digs)
        other = rng.choice([d for d in range(1, 4) if d != b])
        opts = [b, other]
        flip = rng.random() < .5
        if flip:
            opts.reverse()
        out.append(("Three digits are shown below. Exactly one digit differs from the other two. Respond with the value of "
                    f"the differing digit.\n\nDigits: {' '.join(map(str, digs))}\n\nA. {opts[0]}\nB. {opts[1]}\n\n"
                    "Reply with EXACTLY ONE CHARACTER -- A or B -- and nothing else.", "B" if flip else "A"))
    return out


def multiplication(n, rng):
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        out.append((f"Compute the following product. Answer with the number alone as the first word of your answer.\n\n{a} * {b} =",
                    str(a * b)))
    return out


def build_controls(seed=0, n=128, mmlu_subjects=("high_school_biology", "world_religions", "college_chemistry", "us_foreign_policy"),
                   mmlu_per_subject=32) -> dict:
    """name -> dict(items=[(text, gold[, tag])], labels=[...] or None for numeric generation)."""
    rng = random.Random(seed)
    t = {
        "word_recall_8": dict(items=word_recall(n, 8, rng), labels=["1", "2"]),
        "word_recall_48": dict(items=word_recall(n, 48, rng), labels=["1", "2"]),
        "language_localizer": dict(items=localizer(n, rng), labels=["1", "2"]),
        "prontoqa": dict(items=prontoqa(n, rng), labels=["1", "2"]),
        "mmlu_binary": dict(items=mmlu(mmlu_per_subject, mmlu_subjects, rng), labels=["1", "2"]),
        "stroop": dict(items=stroop(n, rng), labels=["1", "2"]),
        "msit": dict(items=msit(n, rng), labels=["A", "B"]),
        "multiplication_1digit": dict(items=multiplication(n, rng), labels=None),
    }
    return t
