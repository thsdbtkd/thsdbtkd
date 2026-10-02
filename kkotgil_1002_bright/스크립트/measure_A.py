import numpy as np
from PIL import Image
BG=np.array([0x9a,0xa3,0x9f])
for d in ['1.5','close']:
    for t,tn in [('now','지금'),('v3','3판')]:
        a=np.asarray(Image.open(f'shots/H_A_{t}_{d}.png').convert('RGB')).astype(float)
        obj=np.abs(a-BG).sum(-1)>18
        green=(a[...,1]>a[...,0]+8)&(a[...,1]>=a[...,2])
        fl=obj&~green; m=a[fl].mean(1)
        print(f"샘플A {tn} {d}: 꽃 화소 {fl.sum()} 최고 {a[fl].max():.0f} 중앙 {np.median(m):.0f} 하위5% {np.percentile(m,5):.0f} | 잎 화소 중앙 {np.median(a[obj&green].mean(1)):.0f} 하위5% {np.percentile(a[obj&green].mean(1),5):.0f}")
