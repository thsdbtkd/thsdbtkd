import numpy as np,glob
from PIL import Image
BG=np.array([0x9a,0xa3,0x9f])
for f in sorted(glob.glob('shots/M_A_*.png')):
    a=np.asarray(Image.open(f).convert('RGB')).astype(float); obj=np.abs(a-BG).sum(-1)>18
    green=obj&(a[...,1]>a[...,0]+8)&(a[...,1]>=a[...,2]); mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/np.maximum(mx,1)
    white=obj&~green&(sat<0.10)&(mx>120); w=a[white]; lf=a[green]
    print(f"{f}: 흰꽃 화소 {white.sum()} 중앙 {np.median(w.mean(1)):.0f} 하위5% {np.percentile(w.mean(1),5):.0f} 최고 {w.max():.0f} R-B {np.median(w[:,0]-w[:,2]):.0f} | 잎 중앙 RGB {tuple(np.median(lf,0).round(0).astype(int).tolist())}" if white.sum() else f"{f}: 흰꽃 없음 | 잎 중앙 {tuple(np.median(lf,0).round(0).astype(int).tolist())}")
