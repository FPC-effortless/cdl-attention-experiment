"""Scientific candidate adapter v2, NOT the frozen raw-state calibration adapter.
Every record exposes its effective payload; original type/reference are retained.
No target annotations or chosen answer are inputs. No trained head is guaranteed
correct by this supplied-state control. Resolution is charged in feature calls.
"""
import time
import torch
from data_adapter import ExactState,CAPACITY,CONDITION_DIM
ADAPTER_VERSION='supplied-resolved-all-records-392-v2'

class ResolvedExactState(ExactState):
    def __init__(self):
        super().__init__();self.resolution_calls=0;self.resolution_hops=0
        self.feature_calls=0;self.feature_seconds=0.
    def resolve(self,identity):
        seen=set();hops=0
        while identity in self.records:
            if identity in seen:return None,hops
            seen.add(identity);hops+=1
            kind,value=self.records[identity]
            if kind=='value':return value,hops
            if kind!='ref':raise ValueError('invalid descriptor')
            identity=value
        return None,hops
    def feature(self,device='cpu'):
        started=time.perf_counter();x=torch.zeros(1,CONDITION_DIM,device=device);stride=22
        for i in range(CAPACITY):
            entry=self.records.get(i);offset=i*stride
            x[0,offset+(0 if entry is None else 1 if entry[0]=='value' else 2)]=1
            value,hops=self.resolve(i)
            self.resolution_calls+=1;self.resolution_hops+=hops
            if value is not None:
                if len(value)!=3:raise ValueError('three-byte payload contract')
                for j,b in enumerate(value):x[0,offset+3+j]=b/255.
            if entry is not None and entry[0]=='ref':x[0,offset+6+entry[1]]=1
        if self.query is not None:x[0,CAPACITY*stride+self.query]=1
        # Synchronize so GPU copy/enqueue work is not hidden from adapter cost.
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        self.feature_calls+=1;self.feature_seconds+=time.perf_counter()-started
        return x
    def receipt(self):
        return {'adapter_version':ADAPTER_VERSION,'feature_calls':self.feature_calls,'resolved_records':self.resolution_calls,'reference_resolution_hops':self.resolution_hops,'feature_seconds':self.feature_seconds,'scope':'all16record resolutions and tensor construction; includes CUDA synchronization; no target metadata'}
