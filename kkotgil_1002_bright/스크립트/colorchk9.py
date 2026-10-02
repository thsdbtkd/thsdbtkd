import numpy as np,glob
from PIL import Image
from scipy.ndimage import binary_erosion
for f in sorted(glob.glob('shots/R_*_v9_*.png')):
    a=np.asarray(Image.open(f).convert('RGB')).astype(int)
    mw=binary_erosion(np.asarray(Image.open(f.replace('R_','mW_')).convert('L'))>100,iterations=1); w=a[mw]
    mf=binary_erosion(np.asarray(Image.open(f.replace('R_','mF_')).convert('L'))>100,iterations=1); px=a[mf]
    pk=px[(px[:,0]-px[:,1]>18)]
    print(f"{f.split('/')[-1]}: 흰 꽃(흰·아이보리·크림 송이) 화소 {len(w)} 평균 RGB {tuple(w.mean(0).round(0).astype(int).tolist())} G>R−3 비율 {(w[:,1]>w[:,0]-3).mean()*100:.1f}% | 분홍·피치 화소 {len(pk)} B>G 비율 {(pk[:,2]>pk[:,1]).mean()*100:.1f}%")
