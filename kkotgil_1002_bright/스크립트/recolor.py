import numpy as np,sys
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from seg import seg
from glbio import save
def lin(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
ROSE=[('white',(240,239,234),.22),('white_cool',(236,239,238),.12),('ivory',(242,234,214),.20),('cream',(243,231,206),.10),
      ('blush',(242,221,212),.10),('peach',(245,213,190),.14),('pink',(243,208,212),.12)]
HYD=[('greenwhite',(232,240,224),.35),('white',(238,238,232),.25),('paleblue',(214,224,240),.15),('blush',(240,222,228),.15),('lime',(226,236,204),.10)]
LEAF=(104,148,68)
def run(f,out,seed,AO=0.30):
    rng=np.random.default_rng(seed)
    S=seg(f); d=S['d']; lab=S['lab']; ch=S['ch']; sz=S['sizes']; cen=S['cen']
    col=np.clip(d['col'],0,1); lum=col@np.array([0.2126,0.7152,0.0722])
    leaf=(ch>=60)&(ch<170); hyd=(~leaf)&(sz==6); rose=(~leaf)&(sz>=8); small=(~leaf)&(sz<6)
    bloom=np.full(S['k'],-1); kind=np.zeros(0,int); B=0; kinds=[]
    for m,per,kd in [(rose,30,0),(hyd,100,1)]:
        idx=np.where(m)[0]; k=max(1,len(idx)//per)
        _,l=kmeans2(cen[idx],k,minit='++',seed=seed,iter=30)
        bloom[idx]=l+B; kinds+= [kd]*k; B+=k
    kinds=np.array(kinds)
    # small bits join nearest bloom
    has=np.where(bloom>=0)[0]; t=cKDTree(cen[has]); sm=np.where(small)[0]
    if len(sm): bloom[sm]=bloom[has[t.query(cen[sm])[1]]]
    # palette per bloom
    tgt=np.zeros((B,3))
    for b in range(B):
        pal=ROSE if kinds[b]==0 else HYD
        w=np.array([p[2] for p in pal]); p=pal[rng.choice(len(pal),p=w/w.sum())]
        c=np.array(p[1],float)*rng.uniform(0.965,1.0)+rng.normal(0,1.5,3)
        tgt[b]=lin(np.clip(c,0,240))
    vb=bloom[lab]; out_col=np.zeros_like(col)
    # shading factor per bloom (normalize by bloom p95)
    flower=vb>=0
    p95=np.zeros(B)
    for b in range(B):
        m=vb==b; p95[b]=np.percentile(lum[m],90) if m.any() else 1
    s=np.clip(lum/np.maximum(p95[np.maximum(vb,0)],1e-4),0,1)
    out_col[flower]=tgt[vb[flower]]*(AO+(1-AO)*s[flower])[:,None]
    # leaves
    lv=~flower; lp=np.percentile(lum[lv],95); sl=np.clip(lum[lv]/lp,0,1)
    leafc=np.array(LEAF,float); comp_j=rng.normal(0,6,(S['k'],3)); comp_j[:,1]*=1.3
    lc=lin(np.clip(leafc+comp_j[lab[lv]],0,242))
    out_col[lv]=lc*(0.35+0.65*sl)[:,None]
    # crevice shade between blooms (cheap AO, kept but weak)
    P=d['pos']; mn,mx=P.min(0),P.max(0); core=np.stack([np.clip(P[:,0],mn[0]+0.3*(mx[0]-mn[0]),mx[0]-0.3*(mx[0]-mn[0])),np.full(len(P),mn[1]),np.full(len(P),(mn[2]+mx[2])/2)],1)
    nd=P-core; nd/=np.linalg.norm(nd,axis=1,keepdims=True)+1e-9
    sub=P[::3]; T=cKDTree(sub); R=0.09*np.linalg.norm(mx-mn)/1.8
    occ=np.zeros(len(P))
    for i,nb in enumerate(T.query_ball_point(P,R,workers=-1)):
        if nb:
            v=sub[nb]-P[i]; occ[i]=(v@nd[i]>0.35*np.linalg.norm(v,axis=1)).sum()
    occ=np.clip(occ/np.percentile(occ,95),0,1)
    out_col*=(1-0.30*occ)[:,None]
    save(d,out_col,out)
    srgb=lambda x: np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
    fl=srgb(out_col[flower]); print(out,'blooms',B,'rose',(kinds==0).sum(),'hyd',(kinds==1).sum(),'flower sRGB max',fl.max().round(1),'p50',np.percentile(fl.mean(1),50).round(1),'p5',np.percentile(fl.mean(1),5).round(1))
    return bloom,kinds,S
if __name__=='__main__':
    run('vr27n_D_bank.glb','web/vr27n_D_bank_bright.glb',7)
    run('vr27n_D_mound_c.glb','web/vr27n_D_mound_c_bright.glb',11)
