import numpy as np,sys,re
from PIL import Image
f=sys.argv[1]; mk=re.sub(r'shots/S_(bank|mound_c)_(\w+?)_(eye|mid|close)\.png',r'shots/mF_\1_\2_\3.png',f)
a=np.asarray(Image.open(f).convert('RGB')).astype(float); m=np.asarray(Image.open(mk).convert('L'))>100
# 경계 화소 제외(마스크 침식)
from scipy.ndimage import binary_erosion
m=binary_erosion(m,iterations=1)
px=a[m]; br=px.mean(1)
print(f"{f.split('/')[-1]}: 꽃 화소 {m.sum()} 평균 {br.mean():.0f} | 밝은 면(상위10%) {np.percentile(br,95):.0f}·평균 {br[br>=np.percentile(br,90)].mean():.0f} | 안쪽 그늘(하위10%) 평균 {br[br<=np.percentile(br,10)].mean():.0f} | 비 {br[br>=np.percentile(br,90)].mean()/br[br<=np.percentile(br,10)].mean():.2f} | 최고 {px.max():.0f}")
