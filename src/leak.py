import sys,numpy as np,pandas as pd,warnings,json;warnings.filterwarnings("ignore")
from exp import oof,ALL,ci
df=pd.read_csv("data/win60.csv");k=sys.argv[1];R=20
n=[oof(df,ALL,k,s,False)[1] for s in range(R)];l=[oof(df,ALL,k,s,True)[1] for s in range(R)]
d=np.array(n)-np.array(l)
json.dump(dict(naive=n,lf=l),open(f"leak_{k}.json","w"))
print(f"{k} naive {ci(n)} lf {ci(l)} infl {ci(d)} pos {100*(d>0).mean():.0f}%")
