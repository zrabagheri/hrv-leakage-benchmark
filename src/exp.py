import numpy as np, pandas as pd, warnings, sys, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
ALL=["mean_hr","sdnn","rmssd","pnn50","cvnn","sd2","sd1_sd2","lf","hf","lf_hf","tp"]
HRV=["sdnn","rmssd","pnn50","sd2","sd1_sd2","lf","hf","lf_hf","tp"]
HR=["mean_hr"]
def M(k):
    return {"LR":Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=3000,class_weight="balanced"))]),
            "RF":RandomForestClassifier(n_estimators=300,max_depth=8,class_weight="balanced",random_state=0,n_jobs=-1),
            "GB":GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=0)}[k]
def oof(df,cols,k,seed,grouped=True):
    X=np.log1p(df[cols].clip(lower=0).values) if False else df[cols].values
    y=df.label.values; g=df.participant.values; p=np.zeros(len(y)); aucs=[]
    sp=StratifiedGroupKFold(5,shuffle=True,random_state=seed) if grouped else StratifiedKFold(5,shuffle=True,random_state=seed)
    for tr,te in sp.split(X,y,g):
        m=M(k); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]; aucs.append(roc_auc_score(y[te],p[te]))
    return p,np.mean(aucs)
def ci(a): a=np.array(a); return f"{a.mean():.3f} [{np.percentile(a,2.5):.3f}-{np.percentile(a,97.5):.3f}]"
def basenorm(df,cols):
    d=df.copy()
    for p,gr in df.groupby("participant"):
        b=gr[gr.task=="Baseline"]
        if len(b)<2: d.loc[gr.index,cols]=np.nan; continue
        d.loc[gr.index,cols]=(gr[cols]-b[cols].mean())/b[cols].std(ddof=1).replace(0,np.nan)
    return d
if __name__!="__main__": sys.exit if False else None
E=sys.argv[1] if len(sys.argv)>1 else "none"; W=int(sys.argv[2]) if len(sys.argv)>2 else 60; R=int(sys.argv[3]) if len(sys.argv)>3 else 20
df=pd.read_csv(f"data/win{W}.csv")
if E=="leak":
    for k in ["LR","RF","GB"]:
        n=[oof(df,ALL,k,s,False)[1] for s in range(R)]; l=[oof(df,ALL,k,s,True)[1] for s in range(R)]
        d=np.array(n)-np.array(l); print(f"{k} naive {ci(n)} lf {ci(l)} infl {ci(d)} pos {100*(d>0).mean():.0f}%",flush=True)
if E=="fs":
    for nm,c in [("All",ALL),("HR only",HR),("HRV only",HRV)]:
        for k in ["LR","RF"]:
            a=[oof(df,c,k,s)[1] for s in range(R)]; pr=[]
            print(f"{nm:<9}{k} AUC {ci(a)}",flush=True)
if E=="task":
    res={}
    for k in ["LR","RF"]:
        out={t:[] for t in ["Stroop","Interview","Hyperventilation"]}; ex=[]
        for s in range(R):
            p,_=oof(df,ALL,k,s)
            for t in out:
                m=(df.label==0)|(df.task==t); out[t].append(roc_auc_score(df.label[m],p[m]))
            d2=df[df.task!="Hyperventilation"].reset_index(drop=True); ex.append(oof(d2,ALL,k,s)[1])
        for t in out: print(f"{k} {t:<17}{ci(out[t])}",flush=True)
        print(f"{k} excl. Hypervent.  {ci(ex)}",flush=True)
if E=="norm":
    nb=df[df.task!="Baseline"].reset_index(drop=True)
    dn=basenorm(df,ALL); dn=dn[dn.task!="Baseline"].dropna(subset=ALL).reset_index(drop=True)
    nb=nb[nb.participant.isin(dn.participant.unique())].reset_index(drop=True)
    print("eval windows",len(nb),len(dn),"participants",dn.participant.nunique())
    for nm,d in [("raw",nb),("baseline-normalised",dn)]:
        for fsn,c in [("All",ALL),("HR only",HR),("HRV only",HRV)]:
            for k in ["LR","RF"]:
                a=[oof(d,c,k,s)[1] for s in range(R)]
                print(f"{nm:<20}{fsn:<9}{k} {ci(a)}",flush=True)
