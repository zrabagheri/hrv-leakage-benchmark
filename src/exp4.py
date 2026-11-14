import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
df=pd.read_csv("data/win60.csv"); df["elapsed"]=(df.start-df.groupby("participant").start.transform("min"))/60
out=[]; pooled=[]
for p,g in df.groupby("participant"):
    g=g.copy(); nonst=g[g.label==0]
    if len(nonst)<4 or g.label.nunique()<2: continue
    # slope of HR over time among non-stress windows (rest + baseline)
    b=np.polyfit(nonst.elapsed,nonst.mean_hr,1)
    g["hr_detr"]=g.mean_hr-np.polyval(b,g.elapsed)
    base=g[g.task=="Baseline"]
    g["hr_norm"]=(g.mean_hr-base.mean_hr.mean())/base.mean_hr.std(ddof=1) if len(base)>1 else np.nan
    ev=g[g.task!="Baseline"]
    if ev.label.nunique()<2: continue
    out.append(dict(p=p,slope=b[0],auc_raw=roc_auc_score(ev.label,ev.mean_hr),auc_detr=roc_auc_score(ev.label,ev.hr_detr),auc_time=roc_auc_score(ev.label,ev.elapsed)))
    pooled.append(ev[["label","hr_detr","hr_norm","elapsed"]])
o=pd.DataFrame(out); P=pd.concat(pooled).dropna()
print("n participants",len(o))
print("HR slope over session among non-stress windows (bpm/min): mean %.3f, median %.3f, pct negative %.0f%%"%(o.slope.mean(),o.slope.median(),100*(o.slope<0).mean()))
print("within-person AUC  raw HR %.3f | detrended HR %.3f | elapsed time %.3f"%(o.auc_raw.mean(),o.auc_detr.mean(),o.auc_time.mean()))
print("pooled AUC (no model)  baseline-normalised HR %.3f | trend-removed HR %.3f | elapsed %.3f"%(roc_auc_score(P.label,P.hr_norm),roc_auc_score(P.label,P.hr_detr),roc_auc_score(P.label,P.elapsed)))
