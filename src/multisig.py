import numpy as np, pandas as pd
BASE="Stress-Predict-Dataset/Raw_data"
def load(p,name):
    path=f"{BASE}/S{p:02d}/{name}.csv"
    with open(path) as f: st=float(f.readline().split(",")[0]); fs=float(f.readline().split(",")[0])
    x=pd.read_csv(path,skiprows=2,header=None).values.astype(float)
    t=st+np.arange(len(x))/fs; return t,x
df=pd.read_csv("data/win60.csv"); out=[]
cache={}
for i,r in df.iterrows():
    p=int(r.participant)
    if p not in cache: cache[p]={n:load(p,n) for n in ["EDA","TEMP","ACC"]}
    ws,we=r.start,r.start+60; f={}
    for n in ["EDA","TEMP"]:
        t,x=cache[p][n]; m=(t>=ws)&(t<we); v=x[m,0]
        f[f"{n.lower()}_mean"]=v.mean() if len(v) else np.nan
        f[f"{n.lower()}_sd"]=v.std() if len(v) else np.nan
        f[f"{n.lower()}_slope"]=np.polyfit(t[m]-ws,v,1)[0] if len(v)>4 else np.nan
    t,x=cache[p]["ACC"]; m=(t>=ws)&(t<we); mag=np.linalg.norm(x[m],axis=1)/64
    f["acc_mean"]=mag.mean() if len(mag) else np.nan; f["acc_sd"]=mag.std() if len(mag) else np.nan
    out.append(f)
m=pd.concat([df,pd.DataFrame(out)],axis=1); m.to_csv("data/win60_multi.csv",index=False)
print(m[["eda_mean","temp_mean","acc_mean"]].describe().round(3)); print("nan rows",m[["eda_mean","temp_mean","acc_mean"]].isna().any(axis=1).sum())
