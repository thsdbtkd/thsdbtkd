import sys,numpy as np
import recolor6_base as R
from glbio import load
from seg import seg
from recolor5 import write_backface,s2l,l2s
from holes2 import band_holes
ROSE_W=[(238,232,218),(240,228,200)]          # 흰 장미, 아이보리 (80%)
ROSE_C=[(240,196,168),(236,190,196)]          # 피치, 연핑크 (20%)
HYD=(226,234,212)
LEAFT=np.array([80,120,55.])
def run(src,seed,GAM=0.8,SPACE='srgb',out=None):
    rng=np.random.default_rng(seed)
    d=load(src+'.glb'); S=seg(src+'.glb'); lab=S['lab']; sz=S['sizes']
    _,flower,vb=R.run(src+'.glb',None,seed,TOP=242,ret=True)   # 송이 나누기만 씀(색은 버림)
    orig=np.clip(d['col'],0,1); Lin=orig@[0.2126,0.7152,0.0722]
    hydv=(sz==6)[lab]
    col=np.zeros_like(orig); blooms=np.unique(vb[flower]); info=[]
    nb_rose=[b for b in blooms if hydv[vb==b].mean()<0.5]
    ncolor=set(rng.choice(nb_rose,int(round(len(nb_rose)*0.2)),replace=False).tolist())
    for b in blooms:
        m=vb==b; ishyd=hydv[m].mean()>=0.5
        if ishyd: t=np.array(HYD,float)
        elif b in ncolor: t=np.array(ROSE_C[rng.integers(2)],float)
        else: t=np.array(ROSE_W[0] if rng.random()<0.6 else ROSE_W[1],float)
        t=t+rng.normal(0,1.5,3)
        L=np.clip(Lin[m]/np.percentile(Lin[m],95),0,1)              # 송이별 상위 95% 로 정규화
        if SPACE=='srgb': col[m]=s2l(np.clip(t[None]*L[:,None]**GAM,0,255))
        else: col[m]=s2l(np.clip(t,0,255))[None]*L[:,None]**GAM
        info.append((b,ishyd))
    # 잎·틈: 원본 밝기(휘도) 그대로, 색만 따뜻한 녹색 쪽으로
    lv=~flower; Tl=s2l(LEAFT); LT=Tl@[0.2126,0.7152,0.0722]
    col[lv]=np.clip(Tl[None]*(Lin[lv]/LT)[:,None],0,1)
    lf=((S['ch']>=60)&(S['ch']<170))[lab]; extra,_=band_holes(d,lf)
    if out: write_backface(d,col,out,extra)
    return col,flower,vb,hydv
if __name__=='__main__':
    GAM=float(sys.argv[1]); SP=sys.argv[2]
    for s,sd in [('vr27n_D_bank',7),('vr27n_D_mound_c',11)]:
        col,flower,vb,hydv=run(s,sd,GAM,SP,'web/'+s+'_bright_v8.glb')
        c=l2s(col); br=c.mean(1); bright=[];inner=[]
        for b in np.unique(vb[flower]):
            m=np.where(vb==b)[0]
            if len(m)<40: continue
            o=np.argsort(br[m]); k=max(1,len(m)//10); bright.append(br[m[o[-k:]]].mean()); inner.append(br[m[o[:k]]].mean())
        bright=np.array(bright); inner=np.array(inner)
        print(f"{s}: 꽃 정점 평균 {br[flower].mean():.0f} | 송이 밝은 면(상위10%) {np.median(bright):.0f} | 안쪽 그늘(하위10%) {np.median(inner):.0f} | 비 {np.median(bright/inner):.2f} | 최고 {c.max():.0f} | 잎 정점 중앙 {tuple(np.median(c[~flower],0).round(0).astype(int).tolist())}")
