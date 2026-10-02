import numpy as np,sys
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from seg import seg
from glbio import save
def lin(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
# 흰색 계열은 아이보리 기운(R-B 4~8)
ROSE=[('white',(240,238,234),.22),('white',(239,237,232),.12),('ivory',(242,234,214),.20),('cream',(243,231,206),.10),
      ('blush',(242,221,212),.10),('peach',(245,213,190),.14),('pink',(243,208,212),.12)]
HYD=[('white',(238,239,232),.35),('white',(240,238,233),.25),('paleblue',(214,224,240),.15),('blush',(240,222,228),.15),('lime',(226,236,204),.10)]
WHITE={'white'}
WRB=(0,2)
WGAIN=1.0  # 따뜻한 빛(R+4,B-4) 적용 전 흰색 R-B, 적용 후 +9~11 → 아래에서 측정
def run(f,out,seed,AO=0.45,CREV=0.30,LEAFGAIN=1.7,TOP=240,PN=68,ret=False):
    rng=np.random.default_rng(seed)
    S=seg(f); d=S['d']; lab=S['lab']; ch=S['ch']; sz=S['sizes']; cen=S['cen']
    col=np.clip(d['col'],0,1); lum=col@np.array([0.2126,0.7152,0.0722])
    leaf=(ch>=60)&(ch<170); hyd=(~leaf)&(sz==6); rose=(~leaf)&(sz>=8); small=(~leaf)&(sz<6)
    bloom=np.full(S['k'],-1); B=0; kinds=[]
    for m,per,kd in [(rose,30,0),(hyd,100,1)]:
        idx=np.where(m)[0]; k=max(1,len(idx)//per)
        _,l=kmeans2(cen[idx],k,minit='++',seed=seed,iter=30)
        bloom[idx]=l+B; kinds+=[kd]*k; B+=k
    kinds=np.array(kinds)
    has=np.where(bloom>=0)[0]; t=cKDTree(cen[has]); sm=np.where(small)[0]
    if len(sm): bloom[sm]=bloom[has[t.query(cen[sm])[1]]]
    tgt=np.zeros((B,3)); names=[]
    for b in range(B):
        pal=ROSE if kinds[b]==0 else HYD
        w=np.array([p[2] for p in pal]); p=pal[rng.choice(len(pal),p=w/w.sum())]
        c=np.array(p[1],float)*rng.uniform(0.975,1.0)+rng.normal(0,1.0,3)
        if p[0] in WHITE:  # 아이보리 기운 유지
            c=c*WGAIN
            c[2]=min(c[2],c[0]-WRB[0]); c[2]=max(c[2],c[0]-WRB[1])
        tgt[b]=lin(np.clip(c,0,TOP)); names.append(p[0])
    names=np.array(names)
    vb=bloom[lab]; flower=vb>=0; out_col=np.zeros_like(col)
    p90=np.ones(B)
    for b in range(B):
        m=vb==b
        if m.any(): p90[b]=np.percentile(lum[m],PN)
    s=np.clip(lum/np.maximum(p90[np.maximum(vb,0)],1e-4),0,1)
    out_col[flower]=tgt[vb[flower]]*(AO+(1-AO)*s[flower])[:,None]
    # 틈 그늘: 꽃잎 정점에만
    P=d['pos']; mn,mx=P.min(0),P.max(0)
    core=np.stack([np.clip(P[:,0],mn[0]+0.3*(mx[0]-mn[0]),mx[0]-0.3*(mx[0]-mn[0])),np.full(len(P),mn[1]),np.full(len(P),(mn[2]+mx[2])/2)],1)
    nd=P-core; nd/=np.linalg.norm(nd,axis=1,keepdims=True)+1e-9
    sub=P[::3]; T=cKDTree(sub); R=0.09*np.linalg.norm(mx-mn)/1.8
    fi=np.where(flower)[0]; occ=np.zeros(len(P))
    for i,nb in zip(fi,T.query_ball_point(P[fi],R,workers=-1)):
        if nb:
            v=sub[nb]-P[i]; occ[i]=(v@nd[i]>0.35*np.linalg.norm(v,axis=1)).sum()
    occ[fi]=np.clip(occ[fi]/np.percentile(occ[fi],95),0,1)
    out_col[fi]*=(1-CREV*occ[fi])[:,None]
    # 잎·초록 띠: 송이별 흔들기 없음, 원본 색에 밝기만 곱함
    lv=~flower; out_col[lv]=np.minimum(col[lv]*LEAFGAIN,lin(TOP))
    if ret: return out_col,flower
    save(d,out_col,out)
    fl=srgb(out_col[flower]).mean(1); wm=flower&np.isin(names[np.maximum(vb,0)],list(WHITE))
    wl=srgb(out_col[wm]); w=wl.mean(1)
    st=dict(max=srgb(out_col[flower]).max(),fl_p50=np.percentile(fl,50),fl_p5=np.percentile(fl,5),
            w_p50=np.percentile(w,50),w_p5=np.percentile(w,5),w_RB=np.median(wl[:,0]-wl[:,2]),
            leaf_p50_before=np.percentile(srgb(col[lv]).mean(1),50),leaf_p50_after=np.percentile(srgb(out_col[lv]).mean(1),50))
    print(out,{k:round(float(v),1) for k,v in st.items()},'blooms',B)
    return st
if __name__=='__main__':
    a=dict(AO=float(sys.argv[1]),CREV=float(sys.argv[2])) if len(sys.argv)>2 else {}
    run('vr27n_D_bank.glb','web/vr27n_D_bank_bright_v2.glb',7,**a)
    run('vr27n_D_mound_c.glb','web/vr27n_D_mound_c_bright_v2.glb',11,**a)
