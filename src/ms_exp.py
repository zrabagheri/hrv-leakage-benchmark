import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
_a=sys.argv[:]; sys.argv=[_a[0]]
from exp import M, ci
sys.argv=_a
G={"HR":["mean_hr"],"EDA":["eda_mean","eda_sd","eda_slope"],"TEMP":["temp_mean","temp_sd","temp_slope"],"ACC":["acc_mean","acc_sd"]}
G["HR+EDA"]=G["HR"]+G["EDA"]; G["All signals"]=G["HR"]+G["EDA"]+G["TEMP"]+G["ACC"]
COLS=sorted(set(sum(G.values(),[])))
df=pd.read_csv("data/win60_multi.csv"); df["elapsed"]=(df.start-df.groupby("participant").start.transform("min"))/60
def norm(df):
    d=df.copy(); keep=[]
    for p,g in df.groupby("participant"):
        b=g[g.task=="Baseline"]
        if len(b)<2: continue
        d.loc[g.index,COLS]=(g[COLS]-b[COLS].mean())/b[COLS].std(ddof=1).replace(0,np.nan); keep+=list(g.index)
    d=d.loc[keep]; return d[d.task!="Baseline"].dropna(subset=COLS).reset_index(drop=True)
raw=df[df.task!="Baseline"].reset_index(drop=True); dn=norm(df); raw=raw[raw.participant.isin(dn.participant.unique())].reset_index(drop=True)
def oofs(d,cols,k,s):
    X=d[cols].values;y=d.label.values;g=d.participant.values;p=np.zeros(len(y));a=[]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=s).split(X,y,g):
        m=M(k);m.fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1];a.append(roc_auc_score(y[te],p[te]))
    return p,np.mean(a)
def tm_auc(d,p,win=10):
    # pairwise AUC over same-participant stress/non-stress pairs within +-win minutes
    num=0;den=0
    for _,g in d.assign(p=p).groupby("participant"):
        s=g[g.label==1];n=g[g.label==0]
        if len(s)==0 or len(n)==0: continue
        dt=np.abs(s.elapsed.values[:,None]-n.elapsed.values[None,:]); ok=dt<=win
        diff=s.p.values[:,None]-n.p.values[None,:]
        num+=((diff>0)+0.5*(diff==0))[ok].sum(); den+=ok.sum()
    return num/den
E=sys.argv[1] if len(sys.argv)>1 else "none"; R=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 5
if E=="sets":
    for nm in ["HR","EDA","TEMP","ACC","HR+EDA","All signals"]:
        for lab,d in [("raw",raw),("cal",dn)]:
            for k in ["LR","RF"]:
                a=[oofs(d,G[nm],k,s)[1] for s in range(R)]
                print(f"{nm:<12}{lab}  {k} {ci(a)}",flush=True)
if E=="drift":
    for var in ["mean_hr","eda_mean","temp_mean","acc_mean"]:
        rr=[];dd=[];sl=[]
        for p,g in df.groupby("participant"):
            ev=g[g.task!="Baseline"];ns=g[g.label==0]
            if ev.label.nunique()<2 or len(ns)<4: continue
            b=np.polyfit(ns.elapsed,ns[var],1); sl.append(b[0]/ns[var].std() if ns[var].std()>0 else 0)
            rr.append(roc_auc_score(ev.label,ev[var])); dd.append(roc_auc_score(ev.label,ev[var]-np.polyval(b,ev.elapsed)))
        rr,dd,sl=map(np.array,(rr,dd,sl))
        print(f"{var:<10} within-AUC raw {rr.mean():.3f} (dir-free {np.mean(np.maximum(rr,1-rr)):.3f}) detrended {dd.mean():.3f} | drift SD/min median {np.median(sl):+.3f} pct same sign {100*max((sl>0).mean(),(sl<0).mean()):.0f}%")
if E=="tm":
    rows=[("Elapsed time",["elapsed"],"LR"),("Elapsed time",["elapsed"],"RF"),("HR (cal)",G["HR"],"LR"),("TEMP (cal)",G["TEMP"],"RF"),("EDA (cal)",G["EDA"],"RF"),("All signals (cal)",G["All signals"],"RF"),("All signals (cal)",G["All signals"],"LR")]
    for nm,c,k in rows:
        st=[];tm=[]
        for s in range(R):
            p,a=oofs(dn,c,k,s); st.append(a); tm.append(tm_auc(dn,p,10))
        print(f"{nm:<18}{k} standard {ci(st)}  time-matched(±10 min) {ci(tm)}",flush=True)
