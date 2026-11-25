"""Paired synthetic three-clause text for a bounded train/held diagnostic."""
from hashlib import sha256
from random import Random
import re

COHORTS = ("train", "cal", "seen", "novel")
MODES = ("bind", "dated", "latest")
HEADS = ("amber", "bright", "cedar", "deep", "east", "frost", "grey", "high")
NOUNS = ("basket", "box", "case", "chest", "crate", "drawer", "pack", "tray")
VALUES = ("blue", "gold", "green", "orange", "pink", "purple", "red", "white")
DATES = ("monday", "tuesday", "wednesday", "thursday")
WRITERS = ("ava", "ben", "cy")
# Offsets partition exact two-word keys; every split has every individual word.
KEYS = {cohort: tuple(f"{h} {NOUNS[(i+c+4*d)%8]}" for i,h in enumerate(HEADS) for d in range(2))
        for c,cohort in enumerate(COHORTS)}
# Train covers prefix or suffix modifiers separately. Only novel combines both
# on one clause or query, and may reverse two known modifier positions.
CLAUSE_BASE = (
    "{who} filed the {key} as {value}{date}",
    "the record says {who} listed the {key} as {value}{date}",
)
QUERY_BASE = {
    "bind": ("what color is the {key}", "what color does the record list for the {key}"),
    "dated": ("what color was the {key} on {when}", "what color does the record list for the {key} on {when}"),
    "latest": ("what color is the latest entry for the {key}", "what color does the latest record list for the {key}"),
}
PREFIX = ("", "in the archive ", "according to the ledger ")
SUFFIX = ("", " in the archive", " according to the ledger")
WORD = re.compile(r"[a-z]+|[.?]")


def _modifier(rng, novel):
    if not novel:
        option = rng.randrange(5)
        return ((option,0) if option<3 else (0,option-2))
    # New compositions of individually available prefix/suffix components.
    return rng.randrange(1,3),rng.randrange(1,3)


def _fact(who,key,value,when,style):
    template, (left,right) = style
    date="" if when is None else f" on {when}"
    return PREFIX[left]+CLAUSE_BASE[template].format(who=who,key=key,value=value,date=date)+SUFFIX[right]


def _ask(mode,key,when,style):
    template, (left,right) = style
    return PREFIX[left]+QUERY_BASE[mode][template].format(key=key,when=when)+SUFFIX[right]+" ?"


def _family(rng,mode,cohort,value_pair):
    novel=cohort=="novel"
    key,other,distractor=rng.sample(KEYS[cohort],3)
    values=tuple(value_pair); dismiss=rng.choice(VALUES)
    dates=tuple(rng.sample(DATES,2)) if mode=="dated" else (None,None)
    keys=(key,other) if mode=="bind" else (key,key)
    # Style and nuisance are constant across four members. Distractor position
    # varies at the family level, independently from writes and query.
    writers=rng.sample(WRITERS,3)
    style=[(rng.randrange(2),_modifier(rng,novel)) for _ in range(4)]
    noise_date=rng.choice(DATES) if mode=="dated" else None
    noise=_fact(writers[2],distractor,dismiss,noise_date,style[2])
    noise_at=rng.randrange(3)
    rows=[]
    for query in range(2):
        assigned=values[::-1] if mode=="latest" and query else values
        facts=[_fact(writers[i],keys[i],assigned[i],dates[i],style[i]) for i in range(2)]
        q=_ask(mode,keys[query] if mode=="bind" else key,dates[query],style[3])
        for reverse in (False,True):
            visible=[1,0] if reverse else [0,1]
            latest=visible[-1]
            visible.insert(noise_at,2)
            selected=latest if mode=="latest" else query
            clauses=facts+[noise]
            text=" . ".join(clauses[i] for i in visible)+" . "+q
            rows.append(dict(text=text,gold=assigned[selected],target=visible.index(selected),mode=mode))
    return rows


def make(seed:int,cohort:str,n:int):
    """n families per mode. Each family is [q0/fwd,q0/rev,q1/fwd,q1/rev].

    In latest, q0/q1 use the same query and swap values while holding the writer
    identities fixed. In bind/dated, q0/q1 select the two keys/dates. Writer
    identities, distractor slot and nuisance contents identify neither answer.
    """
    if type(seed) is not int or type(n) is not int or not 1<=n<=10000 or cohort not in COHORTS:
        raise ValueError("expected integral seed, train/cal/seen/novel and 1..10000 families per mode")
    namespace=int.from_bytes(sha256(f"cog-boundary-001:{cohort}:{seed}".encode()).digest()[:16],"big")
    rng=Random(namespace)
    families=[]; seen=set()
    for mode in MODES:
        colors=list(VALUES); rng.shuffle(colors)
        for i in range(n):
            j=i%len(colors)
            offset=(i//len(colors))%(len(colors)-1)+1
            p=(colors[j],colors[(j+offset)%len(colors)])
            for _ in range(100):
                rows=_family(rng,mode,cohort,p)
                prompts={r["text"] for r in rows}
                if len(prompts)==4 and not prompts&seen:
                    seen.update(prompts)
                    families.append(rows)
                    break
            else:
                raise RuntimeError("could not generate a distinct family")
    return families


def vocab():
    """Sorted exhaustive tokenizer-vocabulary entries, excluding <PAD>."""
    items=set(HEADS+NOUNS+VALUES+DATES+WRITERS+(".", "?", "on"))
    texts=(*CLAUSE_BASE,*(s for qs in QUERY_BASE.values() for s in qs),*PREFIX,*SUFFIX)
    for text in texts:
        items.update(WORD.findall(re.sub(r"\{[^}]+\}","",text)))
    return sorted(items)
