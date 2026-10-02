import numpy as np
from seg import seg
from glbio import load,save
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
TGT=s2l((95,140,70))
# 근접 렌더 잎 중앙이 목표에 오도록 파일별 보정(bank 는 큰 띠 면적 가중, mound_c 는 정점 중앙값)
FAC={'vr27n_D_bank':1.07,'vr27n_D_mound_c':0.97}
AREA={'vr27n_D_bank':True,'vr27n_D_mound_c':False}
for src in ['vr27n_D_bank','vr27n_D_mound_c']:
    S=seg(src+'.glb'); leaf=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    d2=load('web/'+src+'_bright_v2.glb'); out=d2['col'].copy()
    orig=np.clip(S['d']['col'],0,1); L=orig@[0.2126,0.7152,0.0722]
    # 면적 가중 중앙값(화면에 보이는 면적 기준 — bank 의 큰 띠가 목표값에 오게)
    P=S['d']['pos']; t=S['d']['idx'].reshape(-1,3); ar=np.linalg.norm(np.cross(P[t[:,1]]-P[t[:,0]],P[t[:,2]]-P[t[:,0]]),axis=1)/2
    va=np.zeros(len(P)); np.add.at(va,t.ravel(),np.repeat(ar/3,3))
    li=np.where(leaf)[0]; o=np.argsort(L[li]); cw=np.cumsum(va[li][o]); Lm=L[li][o][np.searchsorted(cw,cw[-1]/2)] if AREA[src] else np.median(L[leaf])
    s=np.clip(L[leaf]/Lm,0.25,3.0)       # 원본 명암 비율 = 결
    new=TGT[None,:]*s[:,None]*FAC[src]
    new=np.minimum(new,s2l(220))
    out[leaf]=new
    save(d2,out,'web/'+src+'_bright_v4.glb')
    b=lambda c: l2s(c)
    before=b(d2['col'][leaf]); after=b(out[leaf])
    fl_same=np.abs(d2['col'][~leaf]-out[~leaf]).max()
    print(f"{src} 잎 정점: 2판 중앙 {tuple(np.median(before,0).round(0).astype(int))} 밝기 하위5% {np.percentile(before.mean(1),5):.0f} 최고 {before.max():.0f} → 4판 중앙 {tuple(np.median(after,0).round(0).astype(int))} 밝기 중앙 {np.median(after.mean(1)):.0f} 하위5% {np.percentile(after.mean(1),5):.0f} 최고 {after.max():.0f} | 꽃 정점 변화 최대 {fl_same:.6f}")
