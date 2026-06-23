import sys, statistics
sys.path.insert(0, "bench")
import torch
import build_ext
from benchmark import _cuda_event_time_us, _calibrate_inner

ext = build_ext.get_ext()
dev = torch.device("cuda"); NV = 2

def make(bs):
    L=8; T=2*L*bs+4*bs
    vsl=torch.full((bs,),L,dtype=torch.int64,device=dev)
    pl=torch.empty((bs,0),dtype=torch.int64,device=dev); si=torch.zeros((bs,1),dtype=torch.int64,device=dev)
    o=dict(tm=torch.full((T,),True,dtype=torch.bool,device=dev),
           p=torch.full((bs*NV,),-1,dtype=torch.int64,device=dev),
           ri=torch.full((bs*NV,),-1,dtype=torch.int64,device=dev),
           rnt=torch.full((bs*NV,),-1,dtype=torch.int64,device=dev),
           rns=torch.full((bs*NV,),-1,dtype=torch.int64,device=dev))
    return vsl,pl,si,o

def fns(bs):
    vsl,pl,si,o=make(bs)
    return {
      "noop": lambda: ext.build_tree_noop(vsl,2),
      "baseline": lambda: ext.build_tree_baseline(pl,si,vsl,o["tm"],o["p"],o["ri"],o["rnt"],o["rns"],1,1,2,0),
      "candidate": lambda: ext.build_tree_candidate(pl,si,vsl,o["tm"],o["p"],o["ri"],o["rnt"],o["rns"],1,1,2,0),
    }

def measure(fn, trials=31):
    for _ in range(50): fn()          # warmup BEFORE calibrate (clock up, avoid cold-start)
    torch.cuda.synchronize()
    inner=_calibrate_inner(fn, inner_min=1, inner_max=4096, target_sample_us=1000.0)
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s=sorted(_cuda_event_time_us(fn,inner)[0] for _ in range(trials))
    return statistics.median(s), s[int(0.1*(len(s)-1))], s[int(0.9*(len(s)-1))], inner

print(f"{'bs':>3} {'floor':>7} {'base':>7} {'b_p10':>6} {'b_p90':>6} {'cand':>7} {'c_p10':>6} {'c_p90':>6} {'spd':>5} {'clean_win':>9}")
geo=[]
for bs in (1,2,3,4,5,6,8,10):
    f=fns(bs); r={}
    for n in ("noop","baseline","candidate"): r[n]=measure(f[n])
    fl=r["noop"][0]; bm,bp10,bp90=r["baseline"][0],r["baseline"][1],r["baseline"][2]
    cm,cp10,cp90=r["candidate"][0],r["candidate"][1],r["candidate"][2]
    spd=bm/cm; clean = cp90 < bp10  # candidate strictly faster, non-overlapping
    geo.append(spd)
    print(f"{bs:>3} {fl:>7.3f} {bm:>7.3f} {bp10:>6.3f} {bp90:>6.3f} {cm:>7.3f} {cp10:>6.3f} {cp90:>6.3f} {spd:>5.3f} {str(clean):>9}")
import math
print(f"geomean speedup (bs 1..10) = {math.exp(sum(math.log(x) for x in geo)/len(geo)):.4f}")
