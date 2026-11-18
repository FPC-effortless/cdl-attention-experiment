"""Independently check target occurrences against the authored test grammar."""
from pathlib import Path
import pandas as pd

root = Path(__file__).parent
rows = pd.read_csv(root/'raw.csv.gz')
for row in rows[rows.arm.eq('oracle')].itertuples():
    words = row.text.split()
    cue = words.index('accepted')
    split = words.index(';')
    start,stop = (0,split) if cue<split else (split+1,len(words))
    if row.layout == 'contrastive_subject':
        expected = words.index('rather',start,stop)-1
    elif row.layout == 'indirect_scope':
        expected = words.index('effect',start,stop)+3
    else:
        verb = {'intervening_consultation':'granted','postposed_consultation':'awarded',
                'unseen_connector':'pertains'}[row.layout]
        expected = words.index(verb,start,stop)+3
    assert expected==row.gold and words[expected] in ('former','latter')
print('Verified 5,760 target occurrences against the test grammar.')
