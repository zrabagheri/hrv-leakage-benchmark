import numpy as np,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
labs=["Heart rate","EDA","Skin\ntemperature","Elapsed time"]
fol=[.833,.463,.328,1.0];pre=[.791,.653,.550,0.0]
x=np.arange(4);f,ax=plt.subplots(figsize=(5.6,3.4))
ax.bar(x-.18,fol,.36,color="#1f3a5f",label="Task vs following relaxation")
ax.bar(x+.18,pre,.36,color="#d9822b",label="Task vs preceding relaxation")
for i in range(4):
    ax.text(x[i]-.18,fol[i]+.02,f"{fol[i]:.2f}",ha="center",fontsize=8);ax.text(x[i]+.18,pre[i]+.02,f"{pre[i]:.2f}",ha="center",fontsize=8)
ax.axhline(.5,color="grey",ls=":",lw=1);ax.set_xticks(x);ax.set_xticklabels(labs);ax.set_ylim(0,1.12);ax.set_ylabel("Within-person AUC")
ax.legend(frameon=False,fontsize=8,loc="upper left",ncol=2)
f.savefig("figures/Figure5.png",dpi=300,bbox_inches="tight");f.savefig("figures/Figure5.tif",dpi=300,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
