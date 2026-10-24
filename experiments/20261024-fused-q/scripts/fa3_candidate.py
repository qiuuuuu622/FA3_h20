"""Narrow experimental C ABI for isolated SM90 D256/page16 FA3 kernels."""
import ctypes,pathlib,torch
class FA3Candidate:
    def __init__(self,q,k,v,qs,ks,vs,nk,promote=1,root='/root/gdn35/20261024-fp8-accuracy',variant=None):
        assert q.dtype==k.dtype==v.dtype==torch.float8_e4m3fn
        assert q.shape[1:]==(16,256) and k.shape[1:]==v.shape[1:]==(16,2,256)
        assert k.shape==v.shape and 0<len(q)<=nk<=len(k)*16
        assert all(t.is_cuda and t.is_contiguous() and t.device==q.device for t in [q,k,v,qs,ks,vs])
        assert all(t.shape==(1,2) and t.dtype==torch.float32 for t in [qs,ks,vs])
        assert torch.cuda.get_device_capability(q.device)==(9,0)
        assert torch.cuda.get_device_properties(q.device).multi_processor_count==78
        self.q,self.k,self.v,self.qs,self.ks,self.vs,self.nk=q,k,v,qs,ks,vs,nk
        self.pt=torch.arange(len(k),device=q.device,dtype=torch.int32)
        self.sem=torch.tensor([0,nk],device=q.device,dtype=torch.int32)
        self.y=torch.empty_like(q,dtype=torch.bfloat16);self.lse=torch.empty((16,len(q)),device=q.device)
        name=variant or f'fa3_promote{promote}.so'
        self.lib=ctypes.CDLL(str(pathlib.Path(root)/'fa3_candidate'/name))
        self.lib.fp8_run.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int]*2+[ctypes.c_void_p]
        self.lib.fp8_run.restype=ctypes.c_int
    def __call__(self):
        ts=[self.q,self.k,self.v,self.y,self.lse,self.qs,self.ks,self.vs,self.pt,self.sem]
        code=self.lib.fp8_run(*[x.data_ptr() for x in ts],len(self.q),self.nk,torch.cuda.current_stream(self.q.device).cuda_stream)
        if code:raise RuntimeError(f'cuda error {code}')
        return self.y
