"""Same paired-outcome loss in all arms; prompt and answer sequence are distinct."""
import re
from collections import Counter
from dataclasses import dataclass
import torch
from torch.nn import functional as F

SPECIAL = ("<pad>", "<bos>", "<end>", "<word>", "</word>", "<answer>")
_WORDS = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.ASCII)


@dataclass(frozen=True)
class Tokenizer:
    words: tuple

    @classmethod
    def from_train(cls, texts, answer_words, minimum=5):
        count = Counter(t for text in texts for t in _WORDS.findall(text) if t == t.lower())
        # Nonrepeated identities use exactly representable byte pieces, not an UNK symbol.
        # The holdout text is not read to build the tokenizer.
        return cls(tuple(sorted({w for w,n in count.items() if n >= minimum} | set(answer_words))))

    def __post_init__(self):
        if len(set(self.words))!=len(self.words):
            raise ValueError("duplicate vocabulary")
        object.__setattr__(self,"index",{w:i for i,w in enumerate(self.words,start=len(SPECIAL)+256)})

    @property
    def size(self):
        return len(SPECIAL)+256+len(self.words)

    def encode(self, text):
        ids=[1]
        for word in _WORDS.findall(text):
            # Surface rule, not a list of identities: all capitalized words use
            # the same lowercase byte pieces at fit and evaluation.
            lower=word.lower()
            if word in self.index:
                ids.append(self.index[word])
            else:
                ids.extend([3,*(len(SPECIAL)+i for i in lower.encode("utf8")),4])
        ids.append(5)
        return ids


def paired_loss(logits, lengths, answers, *, lm_targets=None, alpha=1.0, prefix_ce_weight=1.0):
    """answers: [families,2], paired in orientation order; logits see only prefixes."""
    if answers.ndim != 2 or answers.shape[1] != 2 or answers.shape[0]*2 != logits.shape[0]:
        raise ValueError("must supply two worlds per family")
    if torch.any(answers[:,0] == answers[:,1]):
        raise ValueError("paired targets must differ")
    last=logits[torch.arange(logits.size(0),device=logits.device),lengths-1]
    right=last.gather(1,answers.flatten()[:,None]).squeeze(1)
    foil=last.gather(1,answers.flip((1,)).flatten()[:,None]).squeeze(1)
    outcome=F.cross_entropy(last,answers.flatten())
    language=logits.new_zeros(()) if lm_targets is None else F.cross_entropy(
       logits.reshape(-1,logits.size(-1)),lm_targets.reshape(-1),ignore_index=-1)
    return (language if prefix_ce_weight == 1.0 else prefix_ce_weight * language) + outcome + alpha*F.softplus(foil-right).mean(), last
