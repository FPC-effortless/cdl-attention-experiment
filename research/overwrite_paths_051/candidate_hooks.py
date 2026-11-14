"""Unexecuted native adapter candidate; source review is not native qualification.
Requires exact models.py Instrumented from CDL f7b304a. Importing this module
does not import torch or the architecture. Native use requires separate admission.
"""
from contextlib import contextmanager
@contextmanager
def final_set_interventions(model, final_token, *, zero_reads=False, full_gain=False):
    if type(final_token) is not int or final_token not in (2,5):
        raise ValueError('only declared final SET tokens')
    handles=[]
    calls={'zero_reads':0,'full_gain':0}
    def reads(module,args):
        if model.position!=final_token:return None
        x=args[0].clone()
        d=model.config.width
        x[:,d:3*d]=0
        calls['zero_reads']+=1
        return (x,)
    def gain(module,args,output):
        if model.position!=final_token:return None
        calls['full_gain']+=1
        return output.new_full(output.shape,32)
    try:
        if zero_reads:handles.append(model.transform.register_forward_pre_hook(reads))
        if full_gain:handles.append(model.write_gate.register_forward_hook(gain))
        yield calls
        for name,enabled in (('zero_reads',zero_reads),('full_gain',full_gain)):
            if calls[name] != int(enabled):
                raise ValueError('expected one sequence invocation at final SET: '+name)
    finally:
        for h in reversed(handles):h.remove()
def categorical_only(batch):
    # Instrumented.hook expects every key. -100 disables existing gain substitution.
    result=dict(batch)
    result['write_gate']=batch['write_gate'].new_full(batch['write_gate'].shape,-100)
    return result

