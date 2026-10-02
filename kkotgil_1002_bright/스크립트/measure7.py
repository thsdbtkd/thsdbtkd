import numpy as np,sys,glob
from PIL import Image
BG=np.array([0x9a,0xa3,0x9f])
for f in sys.argv[1:]:
    a=np.asarray(Image.open(f).convert('RGB')).astype(float); obj=np.abs(a-BG).sum(-1)>18
    px=a[obj]; br=px.mean(1); mx=px.max(1); sat=(mx-px.min(1))/np.maximum(mx,1)
    w=px[(sat<0.14)&(br>170)]
    print(f"{f.split('/')[-1]}: 전체 화소 하위5% {np.percentile(br,5):.0f} 하위10% {np.percentile(br,10):.0f} 최고 {px.max():.0f} | 흰꽃 평균 RGB {tuple(w.mean(0).round(0).astype(int).tolist()) if len(w) else '-'} 하위5% {np.percentile(w.mean(1),5) if len(w) else 0:.0f} 최고 {w.max() if len(w) else 0:.0f} B최저(하위1%) {np.percentile(w[:,2],1) if len(w) else 0:.0f}")
