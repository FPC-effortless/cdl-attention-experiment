import torch
from .objective import Tokenizer, paired_loss


def test_after_prompt_answer_is_not_inferred_by_input():
    t=Tokenizer.from_train(["aaron handed red to tel"]*6, ["red","blue"])
    assert t.encode("unknownpart") != t.encode("otherunknownpart")
    assert t.encode("unknownpart")[-1]==5
    assert t.index["blue"] not in t.encode("who is the keeper?")


def test_pair_head_rewards_both_directions_and_gradients():
    labels=torch.tensor([[2,3]])
    correct=torch.zeros((2,4,8),requires_grad=True)
    with torch.no_grad():
       correct[0,3,2]=5
       correct[1,2,3]=5
    lengths=torch.tensor([4,3])
    good,last=paired_loss(correct,lengths,labels)
    wrong,_=paired_loss(correct.flip((0,)),lengths,labels)
    assert good<wrong and last.argmax(-1).tolist()==[2,3]
    good.backward()
    assert torch.isfinite(correct.grad).all()


def test_repeated_capitalized_alias_uses_same_pieces_as_unseen_alias():
    t=Tokenizer.from_train(["The clerk filed Ada."]*10,["red","blue"])
    for name in ("Ada","Mia"):
        assert name not in t.index and name.lower() not in t.index
        assert t.encode(name)[1]==3 and t.encode(name)[-2]==4
    assert t.encode("Ada")!=t.encode("Mia")
