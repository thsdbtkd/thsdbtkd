"""10판: 고객 참고 사진(개선된 AR 화면) 색에 맞춤. 명암은 9판 방식(원본 휘도 L, 송이별 상위85%, 지수 1.1)."""
import sys,numpy as np
import recolor6_base as R
from glbio import load
from seg import seg
from recolor5 import write_backface,s2l,l2s
from holes2 import band_holes
# 참고 사진 꽃 부분 k-평균 색에서 고른 팔레트 (색, 송이 비율)
ROSE=[((252,224,216),.30),((254,240,230),.10),((248,196,184),.20),((242,156,178),.28),((232,108,142),.12)]
HYD=[((252,246,236),.60),((224,196,232),.10),((246,202,214),.30)]   # 흰 잔꽃·연분홍·연보라(참고 사진의 라벤더 잔꽃)
LEAF=np.array([128,164,72.])       # 참고 사진 잎(햇살 받은 올리브 연두)
def pick(rng,pal):
    w=np.array([p[1] for p in pal]); return np.array(pal[rng.choice(len(pal),p=w/w.sum())][0],float)
SAT=0.8
def run(src,seed,GAM=1.1,PN=85,LG=1.0,out=None):
    rng=np.random.default_rng(seed)
    d=load(src+'.glb'); S=seg(src+'.glb'); lab=S['lab']; sz=S['sizes']
    _,flower,vb=R.run(src+'.glb',None,seed,TOP=242,ret=True)
    orig=np.clip(d['col'],0,1); Lin=orig@[0.2126,0.7152,0.0722]; hydv=(sz==6)[lab]
    col=np.zeros_like(orig)
    for b in np.unique(vb[flower]):
        m=vb==b; t=pick(rng,HYD if hydv[m].mean()>=0.5 else ROSE)+rng.normal(0,1.5,3)
        L=np.clip(Lin[m]/np.percentile(Lin[m],PN),0,1)
        if hydv[m].mean()>=0.5: L=L**0.55       # 흰 잔꽃(수국 자리)은 그늘을 덜 — 회색으로 안 보이게
        # 그늘이 회색으로 죽지 않게: 어두울수록 같은 색 계열로 진해짐(선형 공간에서 채널별 거듭제곱)
        tl=s2l(np.clip(t,0,255))[None]; Lg=L[:,None]**GAM
        cl=np.power(tl,1+SAT*(1-L[:,None]))*np.power(Lg,1.6)
        c=l2s(cl); c[:,2]=np.minimum(c[:,2],c[:,0]-2)
        col[m]=s2l(np.clip(c,0,252))
    lv=~flower; Ll=Lin[lv]; Lm=np.median(Ll)
    lc=LEAF[None]*np.clip((Ll/Lm)**0.9*LG,0,1.6)[:,None]
    col[lv]=s2l(np.clip(lc,0,235))
    lf=((S['ch']>=60)&(S['ch']<170))[lab]; extra,_=band_holes(d,lf)
    if out: write_backface(d,col,out,extra)
    return col,flower
if __name__=='__main__':
    for s,sd in [('vr27n_D_bank',7),('vr27n_D_mound_c',11)]:
        run(s,sd,out='web/'+s+'_bright_v10.glb')
