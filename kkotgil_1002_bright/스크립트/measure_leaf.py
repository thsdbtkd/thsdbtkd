import numpy as np,sys
from PIL import Image
BG=np.array([0x9a,0xa3,0x9f])
for f in sys.argv[1:]:
    a=np.asarray(Image.open(f).convert('RGB')).astype(float)
    obj=np.abs(a-BG).sum(-1)>18; green=obj&(a[...,1]>a[...,0]+8)&(a[...,1]>=a[...,2])
    px=a[green]; br=px.mean(1)
    import colorsys
    med=np.median(px,0); h=colorsys.rgb_to_hsv(*(med/255))[0]*360
    print(f"{f}: 잎 화소 {green.sum()} 중앙 RGB {tuple(med.round(0).astype(int))} 색상 {h:.0f}° 밝기 중앙 {np.median(br):.0f} 하위5% {np.percentile(br,5):.0f} 최고 {px.max():.0f}")
