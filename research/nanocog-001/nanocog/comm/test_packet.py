"""Exact protocol probes, no fitting or external data."""
import unittest
import torch
from .packet import PacketCommunication

torch.set_num_threads(1)


def compare(a,b):
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)

def same_state(a,b):
    for x,y in zip(a._parts(),b._parts(),strict=True):
        compare(x,y)


class PacketTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.model=PacketCommunication(12,groups=3,topics=3,packet_width=4,capacity=3,local_window=3)
        self.x=torch.randn(3,14,12)

    def test_prefix_future_invariance(self):
        m=self.model
        for training in (True,False):
            m.train(training)
            with torch.no_grad():
                z=self.x.clone()
                z[:,8:]=torch.randn_like(z[:,8:])*11.
                a,sa=m(self.x[:,:8],return_state=True)
                b,sb=m(z[:,:8],return_state=True)
                compare(a,b);same_state(sa,sb)
                compare(a,m(self.x)[:,:8])
                compare(a,m(z)[:,:8])

    def test_cold_packet_read_then_publish(self):
        m=self.model.eval()
        with torch.no_grad():
            m.write_topic.weight.zero_();m.read_topic.weight.zero_()
            m.publish.weight.zero_();m.publish.bias.fill_(10)
            y,s=m(self.x[:,:1],return_state=True)
            yn=m(self.x[:,:1],use_packets=False)
            compare(y,yn)
            assert s.present[0,:,0,0].all()
            p,t=m(self.x[:,1:2],s,return_state=True)
            q=m(self.x[:,1:2],s,use_packets=False)
            assert (p-q).abs().max()>1e-4
            assert t.present[0,:,0,:2].all()

    def test_fifo_replacement_exact(self):
        m=PacketCommunication(2,groups=1,topics=1,packet_width=2,capacity=2,local_window=1).eval()
        x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])
        with torch.no_grad():
            m.payload.weight.copy_(torch.eye(2));m.publish.weight.zero_();m.publish.bias.fill_(10)
            _,s=m(x,return_state=True)
            torch.testing.assert_close(s.packets[0,0,0],torch.tensor([[5.,6.],[3.,4.]]))
            assert s.present.all()

    def test_chunk_single_and_owned(self):
        m=self.model.eval()
        valid=torch.randn(3,14)>-.3
        with torch.inference_mode():
            a,sa=m(self.x,valid=valid,return_state=True)
            b,s=m(self.x[:,:6],valid=valid[:,:6],return_state=True)
            c,sb=m(self.x[:,6:],s,valid=valid[:,6:],return_state=True)
            compare(a,torch.cat((b,c),dim=1));same_state(sa,sb)
            p=m.initial_state(3)
            y=torch.cat([m.step_inplace_(self.x[:,i:i+1],p,valid=valid[:,i:i+1]) for i in range(14)],1)
            compare(a,y);same_state(p,sa)

    def test_padding_is_noop(self):
        m=self.model.eval()
        with torch.inference_mode():
            a,sa=m(self.x[:,:4],return_state=True)
            contaminated=torch.full_like(self.x[:,:2],float("nan"))
            z=torch.cat((self.x[:,:2],contaminated,self.x[:,2:4]),dim=1)
            mask=torch.tensor([[1,1,0,0,1,1]]*3,dtype=torch.bool)
            b,sb=m(z,valid=mask,return_state=True)
            compare(a,b[:,[0,1,4,5]]);same_state(sa,sb)
            assert b[:,2:4].eq(0).all()

    def test_fork_and_reset(self):
        m=self.model.eval()
        with torch.inference_mode():
            _,s=m(self.x[:1,:5],return_state=True)
            backup=s.detached()
            f=s.index_select(torch.tensor([0,0]))
            y=m.step_inplace_(self.x[:2,5:8],f)
            same_state(s,backup)
            for i in range(2):
                z,zs=m(self.x[i:i+1,5:8],s,return_state=True)
                compare(y[i:i+1],z)
                same_state(zs,f.index_select(torch.tensor([i])))
            reset=f.reset_rows(torch.tensor([True,False]))
            same_state(reset.index_select(torch.tensor([0])),m.initial_state(1))
            same_state(reset.index_select(torch.tensor([1])),f.index_select(torch.tensor([1])))

    def test_training_support_and_eval_parity(self):
        m=self.model.train()
        y,s,aux=m(self.x,return_state=True,return_aux=True)
        assert aux['publish_logits'].shape==(3,14,3)
        assert aux['read_topic_logits'].shape==(3,14,3,3)
        y.square().mean().backward()
        for name in ('publish','write_topic','read_topic','payload','local_out','packet_out','packet_gain'):
            d=getattr(m,name).weight.grad
            assert d is not None and torch.isfinite(d).all() and d.abs().sum()>0,name
        with torch.inference_mode():
            y2,s2=m.eval()(self.x,return_state=True)
            compare(y,y2);same_state(s,s2)
            assert s.nbytes==s2.nbytes
            longstate=m(self.x.repeat(1,5,1),return_state=True)[1]
            assert s2.nbytes==longstate.nbytes

    def test_engine_owned_guard_and_capacity_one(self):
        m=PacketCommunication(2,groups=1,topics=1,packet_width=2,capacity=1,local_window=1).eval()
        data=torch.randn(1,3,2);state=m.initial_state(1)
        with self.assertRaises(RuntimeError):m.step_inplace_(data,state)
        with torch.inference_mode():
            _,s=m(data,return_state=True)
            compare(s.positions,torch.tensor([3]))
            empty=m(data[:,:0],s,return_state=True)
            same_state(s,empty[1])


if __name__=='__main__':unittest.main()


def test_empty_functional_branch_owns_separate_storage():
    m = PacketCommunication(8).eval()
    x = torch.randn((2, 3, 8))
    with torch.inference_mode():
        _, original = m(x, return_state=True)
        before = original.detached()
        _, fork = m(x[:, :0], original, return_state=True)
        assert torch.equal(original.positions, fork.positions)
        m.step_inplace_(x, fork)
        for a, b in zip(original._parts(), before._parts(), strict=True):
            assert torch.equal(a, b)
        assert not torch.equal(original.positions, fork.positions)
