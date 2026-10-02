import numpy as np,sys
from PIL import Image,ImageFilter
def floor(w,h,seed=3):
    rng=np.random.default_rng(seed); y,x=np.mgrid[0:h,0:w].astype(float)
    base=np.array([233,216,202.])
    g=(1-0.10*(y/h))[...,None]*base                         # 위가 밝은 그러데이션
    s=np.sin((x*0.9+y*0.55)/38.0)                            # 비스듬한 햇살 줄
    g=g*(1+0.06*np.clip(s,0,1)[...,None])-np.array([6,10,14])*np.clip(-s,0,1)[...,None]
    tile=((x%120)<2)|(((y+0.3*x)%95)<2); g[tile]*=0.93          # 바닥 타일 줄
    return np.clip(g,0,255)
def comp(src,dst):
    a=np.asarray(Image.open(src).convert('RGB')).astype(float); h,w=a.shape[:2]
    bg=np.array([0xeb,0xd8,0xca]); d=np.abs(a-bg).sum(-1)
    m=np.clip(d/40,0,1)[...,None]                            # 가장자리 부드럽게
    f=floor(w,h); out=a*m+f*(1-m)
    Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.3)).save(dst)
if __name__=='__main__': comp(sys.argv[1],sys.argv[2])
