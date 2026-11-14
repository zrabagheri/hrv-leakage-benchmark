import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings("ignore")
_a=sys.argv[:];sys.argv=[_a[0]]
from ms_exp import norm,df as DF
sys.argv=_a
BASE="Stress-Predict-Dataset/Raw_data"
dn=norm(DF)
def pair_auc(pos,neg):
    d=pos[:,None]-neg[None,:]; return ((d>0)+0.5*(d==0)).sum(), d.size
res={}
for var,sign in [("mean_hr",1),("eda_mean",1),("temp_mean",-1),("elapsed",-1)]:
    for direction in ["following","preceding"]:
        num=den=0
        for p,g in dn.groupby("participant"):
            tg=[float(x) for x in open(f"{BASE}/S{int(p):02d}/tags_S{int(p):02d}.csv").read().split()]
            for i in [1,2]:   # interview, hyperventilation (Stroop is preceded by baseline, used for calibration)
                ts,te=tg[2*i],tg[2*i+1]
                task=g[(g.label==1)&(g.start+30>=ts)&(g.start+30<te)]
                if direction=="preceding": rest=g[(g.label==0)&(g.start+30>=tg[2*i-1])&(g.start+30<ts)]
                else:
                    nxt=tg[2*i+2] if len(tg)>2*i+2 else 1e12; rest=g[(g.label==0)&(g.start+30>=te)&(g.start+30<nxt)]
                if len(task) and len(rest):
                    n,d=pair_auc(sign*task[var].values,sign*rest[var].values); num+=n; den+=d
        res[(var,direction)]=num/den
for v in ["mean_hr","eda_mean","temp_mean","elapsed"]:
    print(f"{v:<10} task vs FOLLOWING rest {res[(v,'following')]:.3f} | task vs PRECEDING rest {res[(v,'preceding')]:.3f}")
