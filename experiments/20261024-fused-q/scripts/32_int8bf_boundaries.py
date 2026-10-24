"""INT8 QK / BF16 PV kernel boundary stress (same shapes as 18_mixed_boundaries.py) for compute-sanitizer; also checks vs torch sim."""
import torch,json,os,pathlib,sys
sys.argv=[sys.argv[0],'--out','unused']
src=(pathlib.Path(__file__).parent/'34_int8bf_fusedq_bench.py').read_text().split('paths=sorted')[0]
ns={'__file__':str(pathlib.Path(__file__).parent/'34_int8bf_fusedq_bench.py')};exec(compile(src,'34_int8bf_fusedq_bench.py','exec'),ns)
Int8BF,prep,sim=ns['Int8BF'],ns['prep'],ns['sim']
torch.set_grad_enabled(False);torch.manual_seed(20261024)
lib=os.environ.get('INT8BF_LIB','fa3_int8bf_n96.so')
out=[]
for nq,nk in [(1,1),(1,17),(15,31),(16,32),(17,49),(63,129),(65,513),(129,2049)]:
    q=torch.randn(nq,16,256,device='cuda').bfloat16()
    k=torch.randn((nk+15)//16,16,2,256,device='cuda').bfloat16();v=torch.randn(k.shape,device='cuda').bfloat16()
    Qi,qs,Ki,ks,nq_pad,nk_pad=prep(q,k,nk)
    ctrl=os.environ.get('INT8BF_CTRL')=='1'  # unpatched BF16 build: feed BF16 Q/K, skip sim check
    fq=os.environ.get("INT8BF_FUSEDQ")=="1"  # kernel quantizes BF16 Q itself
    f=Int8BF(q,k,v,qs,ks,nk,nq_pad,nk_pad,lib) if ctrl else Int8BF(q if fq else Qi,Ki,v,qs,ks,nk,nq_pad,nk_pad,lib)
    first=f().clone();torch.cuda.synchronize()
    same=all(torch.equal(first,f()) for _ in range(3));torch.cuda.synchronize()
    sm=sim(Qi,qs,Ki,ks,v,nk);rel=0.0 if ctrl else ((first.float()-sm).norm()/sm.norm()).item()
    out.append(dict(nq=nq,nk=nk,library=lib,finite=bool(torch.isfinite(first).all()),repeat_bitwise=same,rel_vs_sim=rel))
assert all(r['finite'] and r['repeat_bitwise'] and r['rel_vs_sim']<0.01 for r in out),out
print(json.dumps(out),flush=True)
