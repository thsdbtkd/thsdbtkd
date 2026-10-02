import numpy as np,sys
from PIL import Image
def stats(f,bgc=None):
    a=np.asarray(Image.open(f).convert('RGB').resize((340,476))).astype(float); h,w=a.shape[:2]
    c=a[30:h-10, int(w*0.12):int(w*0.88)].reshape(-1,3)
    if bgc is not None: c=c[np.abs(c-np.array(bgc)).sum(1)>30]
    green=(c[:,1]>c[:,0]+8)&(c[:,1]>=c[:,2]); fl=c[~green&(c.mean(1)>140)]; lf=c[green]
    pink=(fl[:,0]-fl[:,1]>40); deep=(fl[:,1]<0.62*fl[:,0])
    return f"꽃 평균 RGB {tuple(fl.mean(0).round(0).astype(int).tolist())} · 분홍 {pink.mean()*100:.1f}% · 진분홍 {deep.mean()*100:.1f}% · 꽃 밝기 중앙 {np.median(fl.mean(1)):.0f} · 잎 중앙 {tuple(np.median(lf,0).round(0).astype(int).tolist())} · 잎 비율 {green.mean()*100:.1f}%"
print('목표:',stats('ref3_right.png'))
for f in sys.argv[1:]: print(f.split('/')[-1]+':',stats(f,(0xeb,0xd8,0xca)))
