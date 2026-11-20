"""Flat baselines and a compositional selective state-transition policy."""
from collections import defaultdict
import math
import re

import numpy as np
from scipy.special import logit
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

C_VALUES = (0.3, 3.0, 30.0)
CLOSE = re.compile(r"\bclos(?:e|ed|ing)\b|\b(?:duplicat\w+|resolved|fixed)\b")
REOPEN = re.compile(r"\bre[ -]?open(?:ed|ing)?\b")
EXPLICIT = re.compile(r"\b(?:i(?:'m| am| will| can)|let's|we (?:can|will)|going to)\b")
HEDGE = re.compile(r"\b(?:if|might|whether|could|should|not|don't|until|maybe)\b")


def cues(row):
    text = row["text"][:12000]
    event = (REOPEN if row["closed"] else CLOSE).search(text)
    return [bool(event), bool(EXPLICIT.search(text)), bool(HEDGE.search(text)), "?" in text,
            bool(event and EXPLICIT.search(text))]


def history(rows):
    return np.asarray([
        [r["closed"], r["reporter"], bool(r["past_actor_actions"]),
         math.log1p(r["past_actor_actions"]), math.log1p(r["past_actor_comments"]),
         math.log1p(r["past_comments"]), math.log1p(max(0, r["age"])) / 10,
         math.log1p(max(0, r["since_action"])) / 10]
        for r in rows
    ])


def txt_features(texts, fit):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=70000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                           max_features=90000, sublinear_tf=True)
    models = [word.fit(texts[fit]), char.fit(texts[fit])]
    return hstack([m.transform(texts) for m in models], format="csr"), models


def train(xs, y, c):
    return LogisticRegression(C=c, solver="liblinear", max_iter=500).fit(xs, y)


def fit(xs, y, train_idx, validation):
    candidates = []
    for c in C_VALUES:
        model = train(xs[train_idx], y[train_idx], c)
        ap = average_precision_score(y[validation], model.predict_proba(xs[validation])[:, 1])
        candidates.append((ap, -c, model))
    return max(candidates, key=lambda m: m[:2])[-1]


def prob(model, xs):
    return model.predict_proba(xs)[:, 1]


def prior_key(row):
    return (row["text"], row["closed"], bool(row["past_actor_actions"]))


def reuse(rows, train_idx, scores):
    support = defaultdict(lambda: defaultdict(set))
    for i in train_idx:
        row = rows[i]
        support[prior_key(row)][row["target"]].add(row["issue"])
    cache = {key: next(iter(outcomes)) for key, outcomes in support.items()
             if len(outcomes) == 1 and len(next(iter(outcomes.values()))) >= 3}
    found = np.asarray([prior_key(row) in cache for row in rows])
    prediction = scores.copy()
    prediction[found] = [
        0.9999 if cache[prior_key(rows[i])] else 0.0001
        for i in np.flatnonzero(found)
    ]
    return prediction, found, cache


def run_models(rows, semantic, train_idx, validation, arm, seed):
    texts = np.asarray([row["text"][:12000] for row in rows], dtype=object)
    y = np.asarray([row["target"] for row in rows])
    groups = np.asarray([row["issue"] for row in rows])
    cue = np.asarray([cues(row) for row in rows], dtype=float)
    context = history(rows)
    text, vocab = txt_features(texts, train_idx)
    extra = np.column_stack([cue, semantic])
    prior = fit(context, y, train_idx, validation)
    lexical = fit(text, y, train_idx, validation)
    checkpoint = {"text_vectors": vocab, "history": prior, "lexical": lexical}
    if arm == "control":
        combined = hstack([text, csr_matrix(context)], format="csr")
        same_inputs = hstack([combined, csr_matrix(extra)], format="csr")
        contextual = fit(combined, y, train_idx, validation)
        flat = fit(same_inputs, y, train_idx, validation)
        checkpoint.update(contextual=contextual, flat=flat)
        return {
            "train_prevalence": np.full(len(rows), y[train_idx].mean()),
            "explicit_rule": cue[:, 4] * (1 - cue[:, 2]) * (1 - cue[:, 3]),
            "zero_shot_nli": semantic[:, 1],
            "history_only": prob(prior, context),
            "lexical": prob(lexical, text),
            "lexical_history": prob(contextual, combined),
            "flat_all_features": prob(flat, same_inputs),
        }, cue, None, checkpoint

    cross_fit = np.empty((len(train_idx), 2))
    folds = GroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for train_local, hold_local in folds.split(train_idx, y[train_idx], groups[train_idx]):
        fit_idx, hold_idx = train_idx[train_local], train_idx[hold_local]
        text_fold, _ = txt_features(texts[np.r_[fit_idx, hold_idx]], np.arange(len(fit_idx)))
        fitted_text = train(text_fold[:len(fit_idx)], y[fit_idx], lexical.C)
        fitted_prior = train(context[fit_idx], y[fit_idx], prior.C)
        cross_fit[hold_local, :] = np.column_stack([
            prob(fitted_text, text_fold[len(fit_idx):]), prob(fitted_prior, context[hold_idx])])
    base = np.column_stack([prob(lexical, text), prob(prior, context)])
    base[train_idx] = cross_fit
    base = logit(np.clip(base, 0.00001, 0.99999))
    modular_features = np.column_stack([base, context[:, 0], extra])
    fused = fit(modular_features, y, train_idx, validation)
    without_semantic = fit(modular_features[:, :-3], y, train_idx, validation)
    scores = prob(fused, modular_features)
    with_reuse, found, cache = reuse(rows, train_idx, scores)
    checkpoint.update(fused=fused, without_semantic=without_semantic, cache=cache)
    return {
        "modular_no_nli": prob(without_semantic, modular_features[:, :-3]),
        "modular": scores,
        "modular_with_reuse": with_reuse,
    }, cue, found, checkpoint
