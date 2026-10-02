import numpy as np,sys
from PIL import Image
# 사용: measure6A.py <이름> <장면png> <수국마스크png> <장미마스크png>
nm,sc,mh,mr=sys.argv[1:5]
a=np.asarray(Image.open(sc).convert('RGB')).astype(float)
def st(mp,lab):
    m=np.asarray(Image.open(mp).convert('L'))>100
    if m.sum()==0: return f"{lab} 없음"
    px=a[m]; br=px.mean(1); mx=px.max(1); sat=(mx-px.min(1))/np.maximum(mx,1)
    w=br[sat<0.10]
    return f"{lab} 화소 {m.sum()} 평균 {br.mean():.0f} 중앙 {np.median(br):.0f} 하위1% {np.percentile(br,1):.0f} 최저 {br.min():.0f} 최고 {px.max():.0f}" + (f" | 흰(채도<0.1) 중앙 {np.median(w):.0f} 하위5% {np.percentile(w,5):.0f}" if len(w) else "")
print(nm, st(mh,'수국'),' || ',st(mr,'장미'))
