import os, numpy as np, pandas as pd
from scipy import signal, interpolate
BASE="Stress-Predict-Dataset"
L=pd.read_csv(f"{BASE}/Processed_data/Improved_All_Combined_hr_rsp_binary.csv")
L.columns=["p","hr","rr","t","y"]
TASKS=["Stroop","Interview","Hyperventilation"]
def load_ibi(p):
    path=f"{BASE}/Raw_data/S{p:02d}/IBI.csv"
    st=float(open(path).readline().split(",")[0])
    d=pd.read_csv(path,skiprows=1,header=None,names=["off","ibi"])
    d["t"]=st+d.off; return d
def tags(p):
    return [float(x) for x in open(f"{BASE}/Raw_data/S{p:02d}/tags_S{p:02d}.csv").read().split()]
def feats(ibi_ms, bt):
    d=np.diff(ibi_ms); f={}
    f["mean_hr"]=60000/np.mean(ibi_ms); f["mean_ibi"]=np.mean(ibi_ms)
    f["sdnn"]=np.std(ibi_ms,ddof=1); f["rmssd"]=np.sqrt(np.mean(d**2))
    f["pnn50"]=100*np.mean(np.abs(d)>50); f["cvnn"]=f["sdnn"]/f["mean_ibi"]
    sd1=np.sqrt(np.var(d,ddof=1)/2); sd2sq=2*np.var(ibi_ms,ddof=1)-np.var(d,ddof=1)/2
    f["sd2"]=np.sqrt(sd2sq) if sd2sq>0 else np.nan; f["sd1_sd2"]=sd1/f["sd2"] if f["sd2"] else np.nan
    t=bt-bt[0]; fi=interpolate.interp1d(t,ibi_ms,bounds_error=False,fill_value=(ibi_ms[0],ibi_ms[-1]))
    tu=np.arange(0,t[-1],0.25)
    if len(tu)<16: f.update(lf=np.nan,hf=np.nan,lf_hf=np.nan,tp=np.nan); return f
    x=fi(tu); x=x-x.mean()
    fr,ps=signal.welch(x,fs=4,nperseg=min(256,len(x)))
    tz=getattr(np,"trapezoid",None) or np.trapz
    band=lambda a,b:(lambda m: tz(ps[m],fr[m]) if m.sum()>1 else np.nan)((fr>=a)&(fr<b))
    f["lf"]=band(.04,.15); f["hf"]=band(.15,.4); f["tp"]=band(.0033,.4)
    f["lf_hf"]=f["lf"]/f["hf"] if f["hf"] and f["hf"]>0 else np.nan
    return f
def build(W=60, MINB_PER_MIN=20):
    rows=[]
    for p in sorted(L.p.unique()):
        d=L[L.p==p]; lab=dict(zip(d.t.astype(int),d.y.astype(int)))
        tg=tags(p)
        if len(tg)<6: print("skip tags",p); continue
        try: ibi=load_ibi(p)
        except Exception as e: print("skip ibi",p); continue
        t0,t1=d.t.min(),d.t.max(); ws=t0; k=0
        while ws+W<=t1:
            we=ws+W; ls=[lab.get(s) for s in range(int(ws),int(we))]
            ok = all(l is not None for l in ls) and len(set(ls))==1
            if ok:
                m=(ibi.t>=ws)&(ibi.t<we); ib=ibi.ibi[m].values*1000; bt=ibi.t[m].values
                v=(ib>300)&(ib<2000); ib,bt=ib[v],bt[v]
                if len(ib)>=MINB_PER_MIN*W/60:
                    f=feats(ib,bt); y=ls[0]
                    # task / phase assignment
                    if y==1:
                        mid=(ws+we)/2; task=min(range(3),key=lambda i: 0 if tg[2*i]-2<=mid<=tg[2*i+1]+2 else min(abs(mid-tg[2*i]),abs(mid-tg[2*i+1]))); task=TASKS[task]
                    else:
                        task="Baseline" if we<=tg[0] else "Rest"
                    f.update(participant=p,start=ws,label=y,task=task,n_beats=len(ib)); rows.append(f)
            ws+=W; k+=1
    return pd.DataFrame(rows)
if __name__=="__main__":
    for W in [60,120,300]:
        df=build(W); df.to_csv(f"data/win{W}.csv",index=False)
        print(W,len(df),df.participant.nunique(),round(df.label.mean(),3),df.task.value_counts().to_dict(),"nan rows",df.isna().any(axis=1).sum())
