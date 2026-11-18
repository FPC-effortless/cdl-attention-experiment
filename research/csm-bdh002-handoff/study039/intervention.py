"""Post-hoc diagnostic for the explicit attachment phrase in the failed layout."""
import pandas as pd
from run_study import ROOT, references

p = pd.read_csv(ROOT/'raw.csv.gz')
p = p[p.layout.eq('same_clause') & p.arm.eq('learned')].copy()


def governed(text):
    words = text.split()
    cue = words.index('accepted')
    limit = words.index(';') if cue < words.index(';') else len(words)
    for i in range(cue+1,limit-3):
        if words[i:i+3] == ['applies','to','the'] and words[i+3] in references:
            return i+3
    return -1


p['governed'] = p.text.map(governed)
assert len(p)==3*96*4
assert p.governed.eq(p.gold).all()
p[['seed','episode','reforder','approval','text','gold','ref','governed']].to_csv(ROOT/'intervention.csv',index=False)
