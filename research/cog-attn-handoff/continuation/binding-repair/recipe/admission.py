"""Conservative exact-action agreement for two role models sharing a lexicon."""


def admitted_action(local_action, local_confidence, sequence_action, sequence_confidence, threshold=.8):
    if local_confidence < threshold or sequence_confidence < threshold:
        return None
    return local_action if local_action == sequence_action else None
