import numpy as np,sys
from glbio import load
from seg import seg
def srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
for src in ['vr27n_D_bank','vr27n_D_mound_c']:
    S=seg(src+'.glb'); leafv=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    for tag,f in [('지금',src+'.glb'),('1판','web/'+src+'_bright.glb'),('2판','web/'+src+'_bright_v2.glb')]:
        c=srgb(load(f)['col']); fl=c[~leafv]; m=fl.mean(1)
        sat=(fl.max(1)-fl.min(1))/np.maximum(fl.max(1),1e-3); w=fl[sat<0.07]; wm=w.mean(1)
        lf=c[leafv].mean(1)
        print(f"{src} {tag}: 꽃 최고 {fl.max():.0f} 중앙 {np.median(m):.0f} 하위5% {np.percentile(m,5):.0f} | 흰꽃({len(w)}정점) 최고 {w.max() if len(w) else 0:.0f} 중앙 {np.median(wm) if len(w) else 0:.0f} 하위5% {np.percentile(wm,5) if len(w) else 0:.0f} R-B {np.median(w[:,0]-w[:,2]) if len(w) else 0:.1f} | 잎·띠 중앙 {np.median(lf):.0f}")
