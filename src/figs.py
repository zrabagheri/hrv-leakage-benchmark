import numpy as np,pandas as pd,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
def save(f,n): f.savefig(f"figures/{n}.png",dpi=300,bbox_inches="tight"); f.savefig(f"figures/{n}.tif",dpi=300,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); plt.close(f)
# Fig 1 protocol + HR for one participant
L=pd.read_csv("Stress-Predict-Dataset/Processed_data/Improved_All_Combined_hr_rsp_binary.csv");L.columns=["p","hr","rr","t","y"]
d=L[L.p==5].sort_values("t");t=(d.t-d.t.min())/60
tg=[float(x) for x in open("Stress-Predict-Dataset/Raw_data/S05/tags_S05.csv").read().split()]
f,ax=plt.subplots(figsize=(7,2.6));ax.plot(t,d.hr.rolling(15,center=True).mean(),color="#1f3a5f",lw=1)
for i,nm in enumerate(["Stroop","Interview","Hyperventilation"]):
    a,b=(tg[2*i]-d.t.min())/60,(tg[2*i+1]-d.t.min())/60; ax.axvspan(a,b,color="#d9822b",alpha=.25,lw=0); ax.text((a+b)/2,ax.get_ylim()[1],nm,ha="center",va="bottom",fontsize=8)
ax.axvspan(0,(tg[0]-d.t.min())/60,color="#7a9cc6",alpha=.2,lw=0); ax.text((tg[0]-d.t.min())/120,ax.get_ylim()[1],"Baseline",ha="center",va="bottom",fontsize=8)
ax.set_xlabel("Time since session start (min)");ax.set_ylabel("Heart rate (bpm)");save(f,"Figure1")
# Fig 2 calibration duration
x=["None\n(raw)","2","3","5","All\n(~7)"]
hr=[.701,.754,.771,.790,.809];hrl=[.678,.739,.764,.782,.805];hru=[.735,.768,.782,.806,.822]
al=[.651,.677,.739,.747,.785];all_l=[.633,.653,.729,.733,.769];all_u=[.673,.704,.749,.760,.798]
f,ax=plt.subplots(figsize=(4.8,3.4));xi=np.arange(5)
for v,lo,hi,c,lab,o in [(hr,hrl,hru,"#1f3a5f","Heart rate only",-.05),(al,all_l,all_u,"#d9822b","All 11 features",.05)]:
    ax.errorbar(xi+o,v,yerr=[np.array(v)-lo,np.array(hi)-v],fmt="o-",color=c,capsize=3,label=lab)
ax.set_xticks(xi);ax.set_xticklabels(x);ax.set_xlabel("Personal baseline used for calibration (min)");ax.set_ylabel("Leak-free AUC (mean, 95% interval)")
ax.axhline(.5,color="grey",ls=":",lw=1);ax.set_ylim(.6,.85);ax.legend(frameon=False,loc="lower right");save(f,"Figure2")
# Fig 3 within-person AUC
df=pd.read_csv("data/win60.csv");df["el"]=(df.start-df.groupby("participant").start.transform("min"))/60
rows=[]
for p,g in df.groupby("participant"):
    ev=g[g.task!="Baseline"];ns=g[g.label==0]
    if ev.label.nunique()<2 or len(ns)<4: continue
    b=np.polyfit(ns.el,ns.mean_hr,1);det=ev.mean_hr-np.polyval(b,ev.el)
    rows.append([roc_auc_score(ev.label,ev.mean_hr),roc_auc_score(ev.label,det)])
r=np.array(rows);np.save("within.npy",r)
f,ax=plt.subplots(figsize=(4.2,3.4))
ax.boxplot([r[:,0],r[:,1]],widths=.45,showfliers=False,medianprops=dict(color="#d9822b"))
for i in range(2): ax.scatter(np.random.default_rng(i).normal(i+1,.05,len(r)),r[:,i],s=12,color="#1f3a5f",alpha=.7,zorder=3)
for a,b in r: ax.plot([1,2],[a,b],color="grey",lw=.4,alpha=.5)
ax.axhline(.5,color="grey",ls=":",lw=1);ax.set_xticks([1,2]);ax.set_xticklabels(["Heart rate","Heart rate,\ntime trend removed"]);ax.set_ylabel("Within-person AUC");save(f,"Figure3")
print("within mean",r.mean(0),"median",np.median(r,0),"n",len(r))
# Fig 4 predictors comparison
labs=["All features\n(raw)","HR\n(calibrated)","Elapsed time\n(LR)","Elapsed time\n(RF)","Time + HR\n(LR)","Time + HR\n(RF)"]
v=[.651,.807,.855,.936,.894,.944];lo=[.633,.805,.853,.930,.891,.937];hi=[.673,.809,.858,.944,.896,.948]
col=["#7a9cc6","#1f3a5f","#d9822b","#d9822b","#8c6d46","#8c6d46"]
f,ax=plt.subplots(figsize=(6.4,3.4));ax.bar(range(6),v,color=col,yerr=[np.array(v)-lo,np.array(hi)-v],capsize=3)
for i,val in enumerate(v): ax.text(i,hi[i]+.008,f"{val:.3f}",ha="center",fontsize=8)
ax.set_xticks(range(6));ax.set_xticklabels(labs,fontsize=8);ax.set_ylim(.5,1);ax.axhline(.5,color="grey",ls=":",lw=1);ax.set_ylabel("Leak-free AUC");save(f,"Figure4")
