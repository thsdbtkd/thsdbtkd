import sys,numpy as np
import recolor6_base as R
from glbio import load
from seg import seg
from recolor5 import write_backface,s2l,l2s
from holes2 import band_holes
ROSE_W=[(244,238,226),(244,232,208)]          # 흰 장미, 아이보리 (80%)
ROSE_C=[(244,202,172),(242,196,196)]          # 피치, 연핑크 (20%)
HYD=(238,236,224)
HYDG=(228,236,214)   # 수국 송이의 30%
LEAFT=np.array([80,120,55.])
def run(src,seed,GAM=1.1,SPACE='srgb',out=None,PN=85):
    rng=np.random.default_rng(seed)
    d=load(src+'.glb'); S=seg(src+'.glb'); lab=S['lab']; sz=S['sizes']
    _,flower,vb=R.run(src+'.glb',None,seed,TOP=242,ret=True)   # 송이 나누기만 씀(색은 버림)
    orig=np.clip(d['col'],0,1); Lin=orig@[0.2126,0.7152,0.0722]
    hydv=(sz==6)[lab]
    col=np.zeros_like(orig); blooms=np.unique(vb[flower]); info=[]; cat=np.zeros(len(orig),int)  # 1 흰, 2 연초록 수국, 3 분홍·피치
    nb_rose=[b for b in blooms if hydv[vb==b].mean()<0.5]
    ncolor=set(rng.choice(nb_rose,int(round(len(nb_rose)*0.2)),replace=False).tolist())
    hb=[b for b in blooms if hydv[vb==b].mean()>=0.5]
    hgreen=set(rng.choice(hb,int(round(len(hb)*0.3)),replace=False).tolist()) if hb else set()
    for b in blooms:
        m=vb==b; ishyd=hydv[m].mean()>=0.5
        if ishyd: t=np.array(HYDG if b in hgreen else HYD,float)
        elif b in ncolor: t=np.array(ROSE_C[rng.integers(2)],float)
        else: t=np.array(ROSE_W[0] if rng.random()<0.6 else ROSE_W[1],float)
        t=t+rng.normal(0,1.5,3)
        k=2 if (ishyd and b in hgreen) else (3 if (not ishyd and b in ncolor) else 1); cat[m]=k
        L=np.clip(Lin[m]/np.percentile(Lin[m],PN),0,1)              # 송이별 상위 PN% 로 정규화
        if SPACE=='srgb':
            c=t[None]*L[:,None]**GAM
            if k==1: c[:,1]=np.minimum(c[:,1],c[:,0]-4)            # 흰 꽃: G ≤ R−3 (회녹색 방지)
            if k==3: c[:,2]=np.minimum(c[:,2],c[:,1]-2)            # 분홍·피치: B < G (자줏빛 방지)
            col[m]=s2l(np.clip(c,0,255))
        else: col[m]=s2l(np.clip(t,0,255))[None]*L[:,None]**GAM
        info.append((b,ishyd))
    # 잎·틈: 원본 밝기(휘도) 그대로, 색만 따뜻한 녹색 쪽으로
    lv=~flower; Tl=s2l(LEAFT); LT=Tl@[0.2126,0.7152,0.0722]
    col[lv]=np.clip(Tl[None]*(Lin[lv]/LT)[:,None],0,1)
    lf=((S['ch']>=60)&(S['ch']<170))[lab]; extra,_=band_holes(d,lf)
    if out: write_backface(d,col,out,extra)
    np.save(src+'_cat9.npy',cat)
    return col,flower,vb,hydv
if __name__=='__main__':
    GAM=float(sys.argv[1]); SP=sys.argv[2]
    for s,sd in [('vr27n_D_bank',7),('vr27n_D_mound_c',11)]:
        col,flower,vb,hydv=run(s,sd,GAM,SP,'web/'+s+'_bright_v9.glb')
        c=l2s(col); br=c.mean(1); bright=[];inner=[]
        for b in np.unique(vb[flower]):
            m=np.where(vb==b)[0]
            if len(m)<40: continue
            o=np.argsort(br[m]); k=max(1,len(m)//10); bright.append(br[m[o[-k:]]].mean()); inner.append(br[m[o[:k]]].mean())
        bright=np.array(bright); inner=np.array(inner)
        print(f"{s}: 꽃 정점 평균 {br[flower].mean():.0f} | 송이 밝은 면(상위10%) {np.median(bright):.0f} | 안쪽 그늘(하위10%) {np.median(inner):.0f} | 비 {np.median(bright/inner):.2f} | 최고 {c.max():.0f} | 잎 정점 중앙 {tuple(np.median(c[~flower],0).round(0).astype(int).tolist())}")
