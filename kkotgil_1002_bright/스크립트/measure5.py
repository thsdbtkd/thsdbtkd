import numpy as np,sys
from glbio import load
from seg import seg
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
for src in ['vr27n_D_bank','vr27n_D_mound_c']:
    S=seg(src+'.glb'); leafv=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    for tag,f in [('지금',src+'.glb'),('4판','web/'+src+'_bright_v4.glb'),('5판','web/'+src+'_bright_v5.glb')]:
        c=srgb(load(f)['col']); fl=c[~leafv]
        sat=(fl.max(1)-fl.min(1))/np.maximum(fl.max(1),1e-3); w=fl[sat<0.07]; wm=w.mean(1); lf=c[leafv]
        ws=f"흰꽃({len(w)}) 중앙 {np.median(wm):.0f} 하위5% {np.percentile(wm,5):.0f} 최고 {w.max():.0f} R-B {np.median(w[:,0]-w[:,2]):.1f}" if len(w) else "흰꽃 없음"
        print(f"{src} {tag}: {ws} | 꽃 전체 중앙 {np.median(fl.mean(1)):.0f} 최고 {fl.max():.0f} | 잎 중앙 RGB {tuple(np.median(lf,0).round(0).astype(int).tolist())} 최고 {lf.max():.0f}")
