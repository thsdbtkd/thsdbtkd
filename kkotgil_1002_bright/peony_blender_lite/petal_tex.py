"""작약 꽃잎 텍스처(512x768): 밝기·결만(색은 정점색) — 밑동 쪽 진하게, 가장자리 밝게, 꽃잎맥, 미세 주름, 물결 가장자리(알파)"""
import numpy as np
from PIL import Image, ImageFilter
W,H=512,768; rng=np.random.default_rng(3)
y,x=np.mgrid[0:H,0:W].astype(float); v=1-y/H; u=(x/W-0.5)*2           # v=0 밑동, 1 끝
# 꽃잎 윤곽: 끝이 넓고 둥근 주걱 + 물결
ph=rng.uniform(0,6.28,6)
edge=np.sin(np.pi*np.clip(v,0,1)**0.7)**0.45*(0.55+0.45*v)
wav=1+0.05*np.sin(x/W*6.28*5+ph[0])+0.035*np.sin(x/W*6.28*11+ph[1])
tipcut=v<0.97*wav-0.06*np.abs(u)**2
alpha=((np.abs(u)<edge)&tipcut&(v>0.01)).astype(float)
# 밝기: 밑동 0.55 → 끝 1.0, 가장자리 살짝 밝게
L=0.84+0.16*np.clip(v,0,1)**0.6+0.06*np.clip(np.abs(u)/np.maximum(edge,1e-3)-0.7,0,1)
# 꽃잎맥: 밑동에서 퍼지는 가는 줄
ang=np.arctan2(u*W/2,(y.max()-y)+1)
veins=0.5+0.5*np.cos(ang*70+np.sin(ang*13)*2)
L*=1-0.035*veins**8*(1-v*0.6)
# 미세 주름·얼룩
n=rng.normal(0,1,(H//8,W//8)); n=np.array(Image.fromarray(((n-n.min())/np.ptp(n)*255).astype(np.uint8)).resize((W,H),Image.BICUBIC))/255
L*=0.95+0.10*n
# 역광 림라이트: 꽃잎 윤곽 안쪽 띠를 밝게(얇은 꽃잎이 빛을 머금은 느낌)
from scipy.ndimage import distance_transform_edt
dist=distance_transform_edt(alpha>0.5)
rim=np.exp(-dist/9.0)*(0.4+0.6*v)
L=np.clip(L*0.92+0.10,0,1); L=L+(1-L)*rim*0.85
img=np.dstack([np.clip(L,0,1)*255]*3+[alpha*255]).astype(np.uint8)
im=Image.fromarray(img,'RGBA'); a=im.split()[3].filter(ImageFilter.GaussianBlur(1.2)); im.putalpha(a)
im.save('petal.png'); print('ok')
