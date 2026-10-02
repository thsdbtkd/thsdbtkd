import numpy as np,glob,sys
from PIL import Image
BG=np.array([154,163,159])
for f in sys.argv[1:]:
    a=np.asarray(Image.open(f).convert('RGB')).astype(float); obj=np.abs(a-BG).sum(-1)>18
    green=obj&(a[...,1]>a[...,0]+8)&(a[...,1]>=a[...,2]); lf=a[green]
    fl=a[obj&~green]; deep=(fl[:,0]>120)&(fl[:,1]<0.5*fl[:,0])&(fl[:,2]>fl[:,1])   # 진분홍: G/R<0.5 (진분홍 100/230=0.43, 분홍 146/242=0.60)
    print(f"{f.split('/')[-1]}: 잎 화소 {green.sum()} 중앙 밝기 {np.median(lf.mean(1)):.0f} 중앙 RGB {tuple(np.median(lf,0).round(0).astype(int).tolist())} | 진분홍 화소 비율 {deep.mean()*100:.2f}%")
