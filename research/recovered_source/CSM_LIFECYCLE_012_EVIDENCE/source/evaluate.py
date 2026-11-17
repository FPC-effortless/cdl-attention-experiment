"""Independent full-session scorer. Gold labels never enter the model forward call.

score(model, cases, batch_size=32) -> {'metrics': ..., 'records': ...}
score_dataset(model, conditions, batch_size=32) -> condition-keyed score results.
Records retain every non-noop prediction and all supplied query/terminal states.
"""
from collections import defaultdict
import torch


def canonical_tuple(pos, op, dst, src, value):
    return [int(pos), int(op), int(dst) if op else 0,
            int(src) if op in (2, 3) else 0, int(value) if op == 1 else 0]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_record(r):
    """Reject incomplete evidence instead of allowing vacuous zip/all passes."""
    length = r['length']
    if type(length) is not int or length <= 0:
        raise ValueError('invalid token length')
    def positions(ps, name):
        if not isinstance(ps, list) or any(type(p) is not int or not 0 <= p < length for p in ps):
            raise ValueError(name + ': invalid positions')
        if ps != sorted(set(ps)):
            raise ValueError(name + ': duplicated or unordered positions')
    def state(st):
        if not isinstance(st, list) or len(st) != 4:
            raise ValueError('state must contain four records')
        for cell in st:
            if not isinstance(cell, list) or len(cell) != 2 or any(type(v) is not int for v in cell):
                raise ValueError('malformed state cell')
            k, v = cell
            if not ((k == 0 and v == 0) or (k == 1 and 1 <= v <= 4) or (k == 2 and 0 <= v < 4)):
                raise ValueError('invalid state cell')
    for key in ('gold_nonnoop', 'pred_nonnoop'):
        rows = r[key]
        for e in rows:
            if not isinstance(e, list) or len(e) != 5 or any(type(v) is not int for v in e):
                raise ValueError('invalid action tuple')
            p, op, dst, src, val = e
            if not 1 <= op <= 5 or not 0 <= dst < 4 or not 0 <= src < 4:
                raise ValueError('invalid action fields')
            if (op not in (2, 3) and src != 0) or (op != 1 and val != 0) or (op == 1 and not 1 <= val <= 4):
                raise ValueError('noncanonical action tuple')
        positions([e[0] for e in rows], key)
    for prefix in ('query', 'final_query'):
        ps = r[prefix + '_positions']
        positions(ps, prefix)
        if (prefix == 'final_query' and len(ps) != 4) or not ps:
            raise ValueError('missing queries')
        for suffix in ('targets', 'predictions'):
            vs = r[prefix + '_' + suffix]
            if not isinstance(vs, list) or len(vs) != len(ps):
                raise ValueError(prefix + ': truncated answer vector')
            lo = 0 if suffix == 'targets' else -1
            if any(type(v) is not int or not lo <= v <= 4 for v in vs):
                raise ValueError('invalid answer code')
    if r['final_query_positions'] != r['query_positions'][-4:]:
        raise ValueError('final queries must be last four gold queries')
    for suffix in ('targets', 'predictions'):
        if r['final_query_' + suffix] != r['query_' + suffix][-4:]:
            raise ValueError('inconsistent final query vector')
    if r['query_positions'] != [e[0] for e in r['gold_nonnoop'] if e[1] == 5]:
        raise ValueError('gold query/action disagreement')
    positions(r['terminal_positions'], 'terminal')
    if not r['terminal_positions'] or not {e[0] for e in r['gold_nonnoop']} <= set(r['terminal_positions']):
        raise ValueError('missing gold terminals')
    for key in ('gold_terminal_states', 'pred_terminal_states'):
        if len(r[key]) != len(r['terminal_positions']):
            raise ValueError('truncated terminal state vector')
        for st in r[key]:
            state(st)
    state(r['gold_final_state']); state(r['pred_final_state'])


def metrics_from_records(records):
    """Pure saved-record rescoring: no torch or model execution required here."""
    sums = defaultdict(int)
    for r in records:
        validate_record(r)
        gold = {e[0]: e for e in r['gold_nonnoop']}
        pred = {e[0]: e for e in r['pred_nonnoop']}
        gp, pp = set(gold), set(pred)
        exact = sum(pred.get(p) == e for p, e in gold.items())
        correct_q = [p == g for p, g in zip(r['query_predictions'], r['query_targets'])]
        final_correct = [p == g for p, g in zip(r['final_query_predictions'], r['final_query_targets'])]
        qpos = set(r['query_positions'])
        pqueries = {p for p, e in pred.items() if e[1] == 5}
        sums['cases'] += 1
        sums['tokens'] += r['length']
        sums['gold_events'] += len(gold)
        sums['pred_events'] += len(pred)
        sums['detected_events'] += len(gp & pp)
        sums['exact_events'] += exact
        sums['false_events'] += len(pp - gp)
        sums['missed_events'] += len(gp - pp)
        sums['wrong_events_at_gold'] += sum(p in pred and pred[p] != e for p, e in gold.items())
        sums['token_action_correct'] += r['length'] - len(gp | pp) + exact
        sums['queries'] += len(qpos)
        sums['query_correct'] += sum(correct_q)
        for target, correct in zip(r['query_targets'], correct_q):
            label = 'present' if target else 'absent'
            sums[label + '_queries'] += 1
            sums[label + '_query_correct'] += int(correct)
        sums['query_operation_correct'] += len(qpos & pqueries)
        sums['query_misclassification'] += len(qpos - pqueries)
        sums['writes_at_query'] += sum(pred.get(p, [0, 0])[1] in (1, 2, 3, 4) for p in qpos)
        sums['false_queries'] += len(pqueries - qpos)
        # A fixed one-byte-late query is diagnostic, never credited as correct.
        sums['one_byte_delayed_queries'] += sum(p not in pqueries and p + 1 in pqueries and p + 1 not in qpos for p in qpos)
        sums['final_maps_correct'] += int(all(final_correct))
        sums['all_queries_correct'] += int(all(correct_q))
        final_state_correct = r['pred_final_state'] == r['gold_final_state']
        sums['final_states_correct'] += int(final_state_correct)
        sums['final_records'] += len(r['gold_final_state'])
        sums['final_records_correct'] += sum(p == g for p, g in zip(r['pred_final_state'], r['gold_final_state']))
        terminal = [p == g for p, g in zip(r['pred_terminal_states'], r['gold_terminal_states'])]
        sums['terminals'] += len(terminal)
        sums['terminal_states_correct'] += sum(terminal)
        action_exact = pred == gold
        sums['action_sessions_correct'] += int(action_exact)
        sums['complete_sessions_correct'] += int(action_exact and all(correct_q) and all(terminal) and final_state_correct)
    c = dict(sums)
    metrics = {'counts': c}
    for name, n, d in [
        ('query_accuracy', 'query_correct', 'queries'),
        ('present_query_accuracy', 'present_query_correct', 'present_queries'),
        ('absent_query_accuracy', 'absent_query_correct', 'absent_queries'),
        ('final_map_accuracy', 'final_maps_correct', 'cases'),
        ('all_query_session_accuracy', 'all_queries_correct', 'cases'),
        ('final_state_accuracy', 'final_states_correct', 'cases'),
        ('final_record_accuracy', 'final_records_correct', 'final_records'),
        ('action_session_accuracy', 'action_sessions_correct', 'cases'),
        ('complete_session_accuracy', 'complete_sessions_correct', 'cases'),
        ('token_action_accuracy', 'token_action_correct', 'tokens'),
        ('terminal_state_accuracy', 'terminal_states_correct', 'terminals'),
        ('event_detection_precision', 'detected_events', 'pred_events'),
        ('event_detection_recall', 'detected_events', 'gold_events'),
        ('exact_event_precision', 'exact_events', 'pred_events'),
        ('exact_event_recall', 'exact_events', 'gold_events'),
        ('query_detection_recall', 'query_operation_correct', 'queries'),
    ]:
        metrics[name] = _ratio(sums[n], sums[d])
    return metrics


def score(model, cases, batch_size=32):
    if batch_size < 1:
        raise ValueError('positive batch_size required')
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        raw = case['text'].encode('utf-8')
        if len(raw) != len(case['text']):
            raise ValueError('position schema requires ASCII text')
        groups[len(raw)].append((index, case, list(raw)))
    indexed_records = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for length, items in groups.items():
                for begin in range(0, len(items), batch_size):
                    batch = items[begin:begin + batch_size]
                    x = torch.tensor([i[2] for i in batch], dtype=torch.long, device=device)
                    # Only raw input and freshly initialized owned model state enter forward.
                    logits, answers, state, traces = model.segment(x, model.init(len(batch)), trace=True)
                    if tuple(logits.shape) != (len(batch), length, 18):
                        raise ValueError('expected [B,T,18] logits')
                    if tuple(answers.shape) != (len(batch), length):
                        raise ValueError('expected [B,T] answers')
                    heads = [z.argmax(-1).cpu().tolist() for z in logits.split([6, 4, 4, 4], dim=-1)]
                    # Value head classes 0..3 represent payload codes 1..4.
                    out = answers.cpu().tolist()
                    kinds, payloads = (t.cpu().tolist() for t in traces)
                    final_kind, final_data = (t.cpu().tolist() for t in state[0])
                    for b, (index, case, _) in enumerate(batch):
                        pred = [canonical_tuple(t, heads[0][b][t], heads[1][b][t], heads[2][b][t], heads[3][b][t] + 1)
                                for t in range(length) if heads[0][b][t] != 0]
                        events = case['events']
                        gold = [canonical_tuple(e['pos'], e['op'], e['dst'], e['src'], e['value']) for e in events if e['op']]
                        qp = case['query_positions']; fqp = case['final_query_positions']
                        gold_answers = {e['pos']: e['answer'] for e in events if e['op'] == 5}
                        if len(fqp) != 4 or any(p not in gold_answers for p in fqp):
                            raise ValueError('four final queries required')
                        r = dict(case_id=case['case_id'], family=case['family'], arm=model.arm,
                                 length=length, pred_nonnoop=pred, gold_nonnoop=gold,
                                 pred_emitted_query_answers=[[p[0], out[b][p[0]]] for p in pred if p[1] == 5],
                                 query_positions=qp, query_targets=[gold_answers[p] for p in qp],
                                 query_predictions=[out[b][p] for p in qp],
                                 final_query_positions=fqp, final_query_targets=[gold_answers[p] for p in fqp],
                                 final_query_predictions=[out[b][p] for p in fqp],
                                 terminal_positions=[e['pos'] for e in events],
                                 gold_terminal_states=[e['state_after'] for e in events],
                                 pred_terminal_states=[[[kinds[b][e['pos']][j], payloads[b][e['pos']][j]] for j in range(4)] for e in events],
                                 gold_final_state=case['final_state'],
                                 pred_final_state=[[final_kind[b][j], final_data[b][j]] for j in range(4)])
                        indexed_records.append((index, r))
    finally:
        model.train(was_training)
    records = [r for _, r in sorted(indexed_records)]
    return {'metrics': metrics_from_records(records), 'records': records}


def score_dataset(model, conditions, batch_size=32):
    return {key: score(model, cases, batch_size) for key, cases in conditions.items()}


evaluate = score
