from .data import ANSWERS, generate, model_view
from .objective import Tokenizer
from .geometry import support_row


def test_old_support_cannot_pass_age_gate():
    fs=generate("train",8,91,delays=(8,))
    tok=Tokenizer.from_train(model_view(fs),ANSWERS)
    w=fs[0].worlds[0]
    blocked=support_row(w,tok,retained_blocks=1)
    allowed=support_row(w,tok,retained_blocks=128)
    assert not blocked["any_support_token_in_age_window"]
    assert allowed["all_support_tokens_in_age_window"]
    assert blocked["support_nearest_age"]==allowed["support_nearest_age"]
