import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from exp import M, ALL, HRV, HR, ci
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score
df=pd.read_csv("data/win60.csv")
def norm(df,cols,kmin=None):
    d=df.copy(); keep=[]
    for p,gr in df.groupby("participant"):
        b=gr[gr.task=="Baseline"].sort_values("start")
        if kmin: b=b.iloc[:kmin]
        if len(b)<2: continue
        sd=b[cols].std(ddof=1).replace(0,np.nan)
        d.loc[gr.index,cols]=(gr[cols]-b[cols].mean())/sd; keep+=list(gr.index)
    d=d.loc[keep]; return d[d.task!="Baseline"].dropna(subset=cols).reset_index(drop=True)
def rep(d,cols,k,R=10,seed0=0):
    X=d[cols].values;y=d.label.values;g=d.participant.values;A=[];P=[]
    for s in range(seed0,seed0+R):
        a=[];pr=[]
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=s).split(X,y,g):
            m=M(k);m.fit(X[tr],y[tr]);p=m.predict_proba(X[te])[:,1];a.append(roc_auc_score(y[te],p));pr.append(average_precision_score(y[te],p))
        A.append(a);P.append(pr)
    return np.array(A),np.array(P)
def nb(diff_folds,ntr,nte):
    # Nadeau-Bengio corrected resampled t-test over all folds
    from scipy import stats
    d=diff_folds.ravel(); n=len(d); mu=d.mean(); var=d.var(ddof=1)
    t=mu/np.sqrt((1/n+nte/ntr)*var); p=2*stats.t.sf(abs(t),n-1); return mu,t,p
E=sys.argv[1] if len(sys.argv)>1 else "none"
if E=="calib":
    for kmin in [2,3,5,None]:
        d=norm(df,ALL,kmin)
        for nm,c in [("HR",HR),("All",ALL)]:
            A,_=rep(d,c,"LR",10); print(f"baseline {kmin or 'all'} min  {nm:<4} LR {ci(A.mean(1))}  n_pts {d.participant.nunique()}",flush=True)
if E=="stats":
    raw=df[df.task!="Baseline"].reset_index(drop=True); dn=norm(df,ALL)
    raw=raw[raw.participant.isin(dn.participant.unique())].reset_index(drop=True)
    res={}
    for nm,d,c in [("raw_all",raw,ALL),("raw_hr",raw,HR),("norm_all",dn,ALL),("norm_hr",dn,HR),("norm_hrv",dn,HRV)]:
        A,P=rep(d,c,"LR",10); res[nm]=A; print(f"{nm:<9} AUC {ci(A.mean(1))} PR-AUC {ci(P.mean(1))} prevalence {d.label.mean():.3f}",flush=True)
    n=len(dn); ntr,nte=0.8*n,0.2*n
    for a,b in [("norm_all","raw_all"),("norm_hr","raw_hr"),("norm_hr","norm_all"),("norm_all","norm_hrv")]:
        mu,t,p=nb(res[a]-res[b],ntr,nte); print(f"{a} - {b}: dAUC {mu:+.3f} t={t:.2f} p={p:.4f}")
if E=="loso":
    raw=df[df.task!="Baseline"].reset_index(drop=True); dn=norm(df,ALL)
    raw=raw[raw.participant.isin(dn.participant.unique())].reset_index(drop=True)
    for nm,d,c in [("raw HR",raw,HR),("norm HR",dn,HR),("norm All",dn,ALL)]:
        X=d[c].values;y=d.label.values;g=d.participant.values;a=[]
        for tr,te in LeaveOneGroupOut().split(X,y,g):
            if len(set(y[te]))<2: continue
            m=M("LR");m.fit(X[tr],y[tr]);a.append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
        a=np.array(a); print(f"LOSO {nm:<9} mean {a.mean():.3f} SD {a.std():.3f} median {np.median(a):.3f} n={len(a)} pct>0.5 {100*(a>0.5).mean():.0f}%")
if E=="controls":
    # time-trend control: elapsed time since session start as only predictor
    d=df[df.task!="Baseline"].copy(); d["elapsed"]=d.start-d.groupby("participant").start.transform("min")
    d=d.reset_index(drop=True); A,_=rep(d,["elapsed"],"LR",10); print("time-only control AUC",ci(A.mean(1)))
    # within-participant label permutation (negative control) on normalised HR
    dn=norm(df,ALL); rng=np.random.default_rng(0); A=[]
    for s in range(10):
        dp=dn.copy(); dp["label"]=dp.groupby("participant").label.transform(lambda v: rng.permutation(v.values))
        a,_=rep(dp,HR,"LR",1,s); A.append(a.mean())
    print("permuted-label control (norm HR)",ci(A))
