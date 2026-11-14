import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings("ignore")
_a=sys.argv[:];sys.argv=[_a[0]]
from exp import M,ci
sys.argv=_a
from ms_exp import G,norm,df as DF
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
BASE="Stress-Predict-Dataset/Raw_data"
dn=norm(DF)
# block assignment: task i and the relaxation that follows it
blk=[]
for _,r in dn.iterrows():
    tg=[float(x) for x in open(f"{BASE}/S{int(r.participant):02d}/tags_S{int(r.participant):02d}.csv").read().split()]
    mid=r.start+30; b=-1
    for i in range(3):
        end=tg[2*i+2] if i<2 and len(tg)>2*i+2 else 1e12
        if tg[2*i]-5<=mid<end: b=i
    blk.append(b)
dn["block"]=blk; dn=dn[dn.block>=0].reset_index(drop=True)
print("windows per block",dn.groupby(["block","label"]).size().to_dict())
def run(cols,k,R):
    res=[]
    for s in range(R):
        rng=np.random.default_rng(s); parts=dn.participant.unique(); rng.shuffle(parts)
        folds=np.array_split(parts,5); scores=[]
        for B in [0,1]:   # hold out Stroop block or Interview block (hyperventilation too small)
            p=np.full(len(dn),np.nan)
            for f in folds:
                te=dn.participant.isin(f)&(dn.block==B); tr=(~dn.participant.isin(f))&(dn.block!=B)
                m=M(k);m.fit(dn.loc[tr,cols].values,dn.loc[tr,"label"].values);p[te.values]=m.predict_proba(dn.loc[te,cols].values)[:,1]
            msk=dn.block==B; scores.append(roc_auc_score(dn.label[msk],p[msk.values]))
        res.append(scores)
    return np.array(res)
R=int(_a[1]) if len(_a)>1 else 3
for nm,c,k in [("Elapsed time",["elapsed"],"LR"),("Elapsed time",["elapsed"],"RF"),("HR (cal)",G["HR"],"LR"),("EDA (cal)",G["EDA"],"RF"),("TEMP (cal)",G["TEMP"],"RF"),("All signals (cal)",G["All signals"],"LR"),("All signals (cal)",G["All signals"],"RF")]:
    a=run(c,k,R); print(f"{nm:<18}{k} held-out Stroop block {ci(a[:,0])}  held-out Interview block {ci(a[:,1])}",flush=True)
