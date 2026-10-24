"""INT8 QK (+Hadamard, K-smooth, kgrp K scale, per-token Q scale) / BF16 PV kernel: accuracy + ABBA timing.

Quantization = 26_pv_split_screen.py 'i8had_pbf_vbf' (sim 0.51% vs BF16). Reports kernel error vs the BF16 capture output
and vs the torch sim of the same quantization. Timing: attention kernel only (Q/K quant excluded; Q prep timed separately).
"""
import argparse,ctypes,json,pathlib,statistics,sys
import torch,scipy.linalg
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from fa3_candidate import FA3Candidate
p=argparse.ArgumentParser();p.add_argument('--root',default='/root/gdn35/20261024-fp8-accuracy')
p.add_argument('--glob',default='sample_*_layer*.pt');p.add_argument('--out',required=True)
p.add_argument('--variants',default='fa3_int8bf_n80.so');p.add_argument('--ctrl',default='fa3_int8bf_ctrl_bf16.so')
p.add_argument('--timing',action='store_true');p.add_argument('--reps',type=int,default=5);p.add_argument('--rounds',type=int,default=6)
a=p.parse_args();R=pathlib.Path(a.root);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
from sgl_kernel.flash_attn import flash_attn_with_kvcache as fa3
H=torch.tensor(scipy.linalg.hadamard(256),dtype=torch.float32,device='cuda')/16
def q8_tok(x):
    s=x.abs().amax(-1,keepdim=True).clamp_min(1e-10)/127;return torch.round(x/s).clamp(-127,127),s
def q8_kgrp(x):
    n=len(x);t=torch.arange(n,device=x.device);g=(t//128)*4+(t%8)//2;G=int(g.max())+1
    a_=torch.zeros(G,device=x.device).scatter_reduce(0,g,x.abs().amax(1),'amax');s=(a_.clamp_min(1e-10)/127)[g][:,None]
    return torch.round(x/s).clamp(-127,127),s
class Int8BF:
    def __init__(self,q,k,v,qs,ks,nk,nq_pad,nk_pad,variant):
        assert (q.dtype==k.dtype or 'fusedq' in variant) and v.dtype==torch.bfloat16 and q.shape[1:]==(16,256) and k.shape[1:]==v.shape[1:]==(16,2,256)
        assert all(t.is_cuda and t.is_contiguous() for t in [q,k,v,qs,ks]) and qs.shape==(16,nq_pad) and ks.shape==(2,nk_pad)
        assert torch.cuda.get_device_capability(q.device)==(9,0) and 0<len(q)<=nk<=len(k)*16
        self.a=[q,k,v];self.qs,self.ks,self.nk,self.pads=qs,ks,nk,(nq_pad,nk_pad)
        self.pt=torch.arange(len(k),device=q.device,dtype=torch.int32);self.sem=torch.tensor([0,nk],device=q.device,dtype=torch.int32)
        self.y=torch.empty(q.shape,device=q.device,dtype=torch.bfloat16);self.lse=torch.empty((16,len(q)),device=q.device)
        self.lib=ctypes.CDLL(str(R/'fa3_candidate'/variant))
        self.lib.int8bf_run.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*4+[ctypes.c_void_p];self.lib.int8bf_run.restype=ctypes.c_int
    def __call__(self):
        ts=[*self.a,self.y,self.lse,self.qs,self.ks,self.pt,self.sem]
        code=self.lib.int8bf_run(*[x.data_ptr() for x in ts],len(self.y),self.nk,*self.pads,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(f'cuda error {code}')
        return self.y
def err(x,y):
    x=x.float();y=y.float();d=x-y
    return dict(relative_l2=(d.norm()/y.norm()).item(),max_abs=d.abs().max().item(),close=bool(torch.allclose(x,y,atol=.003,rtol=.01)),
                element_pass_fraction=(d.abs()<=.003+.01*y.abs()).float().mean().item(),finite=bool(torch.isfinite(x).all()))
def prep(q,k,n):
    """int8 Q [nq,16,256], per-token q scale [16,nq_pad]; int8 K paged [P,16,2,256], kgrp k scale per token [2,nk_pad]."""
    nq=len(q);P=len(k);nq_pad=(nq+127)//128*128+128;nk_pad=(P*16+127)//128*128+128
    Qi,qs=q8_tok(q.float()@H);qs_t=torch.zeros(16,nq_pad,device='cuda');qs_t[:,:nq]=qs[...,0].T
    kf=k.reshape(-1,2,256)[:n].float();Ki=torch.zeros(P*16,2,256,device='cuda');ks_t=torch.zeros(2,nk_pad,device='cuda')
    for h in range(2):
        K=kf[:,h]@H;K=K-K.mean(0,keepdim=True);ki,s=q8_kgrp(K);Ki[:n,h]=ki;ks_t[h,:n]=s[:,0]
    return Qi.to(torch.int8).contiguous(),qs_t,Ki.to(torch.int8).reshape(P,16,2,256).contiguous(),ks_t,nq_pad,nk_pad
def sim(Qi,qs_t,Ki,ks_t,v,n):
    """Torch model of the kernel: s32 QK, fp32 scale, BF16 P x BF16 V."""
    nq=len(Qi);out=torch.empty(nq,16,256,device='cuda');mask=torch.arange(n-nq,n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None]
    Kf=Ki.reshape(-1,2,256)[:n].float();V=v.reshape(-1,2,256)[:n].float()
    for h in range(16):
        kh=h//8;s=(Qi[:,h].float()@Kf[:,kh].T)*qs_t[h,:nq,None]*ks_t[kh,None,:n]/16;s.masked_fill_(~mask,-1e30)
        p_=torch.exp(s-s.amax(1,keepdim=True));out[:,h]=(p_.bfloat16().float()@V[:,kh])/p_.sum(1,keepdim=True)
    return out
def quant8(x):
    s=x.float().abs().max().clamp_min(1e-10)/448;return (x.float()/s).to(torch.float8_e4m3fn),s.expand(1,2).contiguous()
paths=sorted(p for p in (R/'captures').glob(a.glob) if 'layer' in p.name);variants=a.variants.split(',')
for path in paths:
    c=torch.load(path,weights_only=True);n=c['meta']['kv_tokens'];q=c['q'].cuda().contiguous();k=c['k_cache'].cuda().contiguous();v=c['v_cache'].cuda().contiguous();nq=len(q)
    Qi,qs_t,Ki,ks_t,nq_pad,nk_pad=prep(q,k,n)
    fns={name.removesuffix('.so'):Int8BF(q if 'fusedq' in name else Qi,Ki,v,qs_t,ks_t,n,nq_pad,nk_pad,name) for name in variants}
    fns['ctrl_bf16']=Int8BF(q,k,v,qs_t,ks_t,n,nq_pad,nk_pad,a.ctrl)
    cu=torch.tensor([0,nq],device='cuda',dtype=torch.int32);seq=torch.tensor([n],device='cuda',dtype=torch.int32);pt=torch.arange(len(k),device='cuda',dtype=torch.int32).reshape(1,-1)
    fns['bf16']=lambda:fa3(q=q,k_cache=k,v_cache=v,page_table=pt,cache_seqlens=seq,cu_seqlens_q=cu,max_seqlen_q=nq,causal=True,softmax_scale=1/16)
    if a.timing:
        q8,qs8=quant8(q);k8,ks8=quant8(k);v8,vs8=quant8(v);fns['fp8_promote1_c128']=FA3Candidate(q8,k8,v8,qs8,ks8,vs8,n,root=a.root,variant='fa3_promote1_c128.so')
    ref=c['output'].cuda();outs={name:fn().clone() for name,fn in fns.items()};torch.cuda.synchronize();sm=sim(Qi,qs_t,Ki,ks_t,v,n)
    row=dict(capture=path.name,kv=n,nq=nq,errors_vs_bf16={name:err(o,ref) for name,o in outs.items()},
             errors_vs_sim={name:err(outs[name],sm) for name in [x.removesuffix('.so') for x in variants]},sim_vs_bf16=err(sm,ref),
             repeat_bitwise={name:all(torch.equal(outs[name],fn()) for _ in range(3)) for name,fn in fns.items()})
    if a.timing:
        fns['q_prep']=lambda:q8_tok(q.float()@H)
        graphs={}
        for name,fn in fns.items():
            for _ in range(3):fn()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(a.reps):fn()
            graphs[name]=g
        times={}
        for r in range(a.rounds):
            for name in (list(graphs) if r%2==0 else list(reversed(graphs))):
                s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                s.record();graphs[name].replay();e.record();e.synchronize();times.setdefault(name,[]).append(s.elapsed_time(e)/a.reps)
        med={k_:statistics.median(v_) for k_,v_ in times.items()}
        row.update(times_ms=times,medians_ms=med,speedup_vs_bf16={k_:med['bf16']/v_ for k_,v_ in med.items()},
                   timing_scope='attention kernel only; K prequantized; Q prequantized except fusedq variants; q_prep = unfused torch Q Hadamard+quant')
    with (R/'results'/a.out).open('a') as f:f.write(json.dumps(row)+'\n')
    print(path.name,json.dumps({n_:round(e_['relative_l2']*100,4) for n_,e_ in row['errors_vs_bf16'].items()}),
          'vs_sim',json.dumps({n_:round(e_['relative_l2']*100,4) for n_,e_ in row['errors_vs_sim'].items()}),
          json.dumps({k_:round(v_,4) for k_,v_ in row.get('medians_ms',{}).items()}),flush=True)
