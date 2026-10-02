import numpy as np,sys
from glbio import load
from seg import seg
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
for src in ['vr27n_D_bank','vr27n_D_mound_c']:
    S=seg(src+'.glb'); leafv=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    for tag,f in [('지금',src+'.glb'),('4판','web/'+src+'_bright_v4.glb'),('6판','web/'+src+'_bright_v6.glb')]:
        c=srgb(load(f)['col']); fl=c[~leafv]
        sat=(fl.max(1)-fl.min(1))/np.maximum(fl.max(1),1e-3); w=fl[sat<0.07]; wm=w.mean(1); lf=c[leafv]
        ws=f"흰꽃({len(w)}) 중앙 {np.median(wm):.0f} 하위5% {np.percentile(wm,5):.0f} 최고 {w.max():.0f} R-B {np.median(w[:,0]-w[:,2]):.1f}" if len(w) else "흰꽃 없음"
        print(f"{src} {tag}: {ws} | 꽃 전체 중앙 {np.median(fl.mean(1)):.0f} 최고 {fl.max():.0f} | 잎 중앙 RGB {tuple(np.median(lf,0).round(0).astype(int).tolist())} 최고 {lf.max():.0f}")
# 송이별 바깥(밝은 쪽 20%) / 안쪽(어두운 쪽 20%) — 흰 송이만
import recolor6_base as R
from recolor6 import depth_shade
for src,seed in [('vr27n_D_bank',7),('vr27n_D_mound_c',11)]:
    d=load(src+'.glb'); S=seg(src+'.glb'); c=srgb(load('web/'+src+'_bright_v6.glb')['col'])
    _,flower,vb=R.run(src+'.glb',None,seed,TOP=242,ret=True)
    f=depth_shade(d['pos'],flower,vb,None,0.3)
    fl=c[flower]; sat=(fl.max(1)-fl.min(1))/np.maximum(fl.max(1),1e-3)
    wv=np.where(flower)[0][sat<0.07]; out=[];inn=[]
    for b in np.unique(vb[wv]):
        m=wv[vb[wv]==b]
        if len(m)<50: continue
        o=np.argsort(f[m]); k=max(1,len(m)//5)
        inn.append(c[m[o[:k]]].mean()); out.append(c[m[o[-k:]]].mean())
    wm=c[wv].mean(1)
    print(f"{src} 6판 흰 송이 {len(out)}개: 바깥(중심) 평균 {np.mean(out):.0f} → 꽃 속 안쪽 평균 {np.mean(inn):.0f} (차 {np.mean(out)-np.mean(inn):.0f}) | 흰꽃 평균 {wm.mean():.0f}")
