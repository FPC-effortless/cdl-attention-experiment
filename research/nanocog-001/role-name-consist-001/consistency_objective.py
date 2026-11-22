"""Eight-row task loss plus a fixed bounded remapped-margin agreement term."""
import torch
from torch.nn import functional as F

COEFFICIENTS=(0.,.25)
SCALE=2.
CONSIST=(0,1,4,5),(2,3,6,7)

def answer_foil_consistency(tail,answers,*,paired,coefficient):
    if tail.ndim!=2 or tail.shape!=(8,287) or answers.shape!=(8,) or answers.dtype!=torch.long:
        raise ValueError('exact eight full-vocabulary answer rows required')
    if type(paired) is not bool or coefficient not in COEFFICIENTS or answers.device!=tail.device:
        raise ValueError('unregistered objective')
    ix=torch.arange(8,device=tail.device)
    foil=answers[ix^1]
    if bool((answers==foil).any()):
        raise ValueError('assignment-flip sibling must have a different answer')
    margin=tail[ix,answers]-tail[ix,foil]
    ans=F.cross_entropy(tail,answers)
    contrast=F.softplus(-margin).mean()
    if paired:
        j,k=(torch.tensor(p,device=tail.device) for p in CONSIST)
        if not (torch.equal(answers[j],answers[k]) and torch.equal(foil[j],foil[k])):
            raise ValueError('remap rows must share true and foil colors')
        compressed=torch.tanh(margin/SCALE)
        consistency=((compressed[j]-compressed[k])*.5).square().mean()
    else:
        consistency=tail.new_zeros(())
    return ans+contrast+coefficient*consistency, dict(answer_ce=ans,assignment_foil=contrast,
         consistency=consistency,signed_margin=margin)
