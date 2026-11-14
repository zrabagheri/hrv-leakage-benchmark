import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from exp import M, ALL, HR, ci
from exp2 import norm, rep, nb
df=pd.read_csv("data/win60.csv")
df["elapsed"]=df.start-df.groupby("participant").start.transform("min")
dn=norm(df,ALL)  # keeps elapsed column
for nm,c,k in [("time LR",["elapsed"],"LR"),("time RF",["elapsed"],"RF"),("normHR LR",HR,"LR"),("time+normHR LR",["elapsed"]+HR,"LR"),("time+normHR RF",["elapsed"]+HR,"RF")]:
    A,_=rep(dn,c,k,5); print(f"{nm:<16}{ci(A.mean(1))}",flush=True)
# time-matched: each task window vs rest windows within +-10 min in same participant
from sklearn.metrics import roc_auc_score
r=[]
for p,g in dn.groupby("participant"):
    pass
