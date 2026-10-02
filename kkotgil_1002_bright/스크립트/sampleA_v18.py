"""샘플 꽃둑 18판: 원본(첫 번째) 샘플 꽃둑의 작약 형상·꽃잎 텍스처 결·광택(sheen)을 그대로 두고 색만 목표 사진처럼.
- 꽃 텍스처: 칸(작약 종류)마다 밝기 결만 남기고(채도 제거, 대비 유지) → 색은 송이별 정점색으로
- 수국 덩어리 제거 → 그 자리에 원본 작약 송이 복제
- 잎: 텍스처 밝히고 올리브·노란 기
- 흰 잔꽃·라벤더 잔꽃, 날리는 꽃잎"""
import io,sys,numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from fbio import G
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def enc(a,mime,alpha):
    buf=io.BytesIO()
    if mime.endswith('jpeg') and not alpha:
        Image.fromarray(np.clip(a[...,:3],0,255).round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=92); return buf.getvalue(),'image/jpeg'
    Image.fromarray(np.clip(a,0,255).round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); return buf.getvalue(),'image/png'
MIXO=0.0
def petal_tex(img,lo,hi):
    a=np.asarray(img.convert('RGBA')).astype(float); m=a[...,3]>10; L=a[...,:3]@[0.299,0.587,0.114]
    h,w=L.shape; out=np.zeros_like(L)
    for gy in range(2):
        for gx in range(2):
            sl=(slice(gy*h//2,(gy+1)*h//2),slice(gx*w//2,(gx+1)*w//2)); mm=m[sl]
            if not mm.any(): continue
            p2,p98=np.percentile(L[sl][mm],[2,98]); out[sl]=np.clip((L[sl]-p2)/max(p98-p2,1),0,1)
    v=lo+(hi-lo)*out**0.85
    # 원본 색 결(꽃 속 진한 분홍)도 일부 남김: 채도를 MIXO 만큼
    orig=a[...,:3]; Lo=orig@[0.299,0.587,0.114]; ratio=orig/np.maximum(Lo,1)[...,None]
    ratio=1+(ratio-1)*MIXO
    a[...,:3]=np.clip(v[...,None]*ratio,0,255); return a
def leaf_tex(img,gain,shift,sat,cap=225):
    import matplotlib.colors as mc
    a=np.asarray(img.convert('RGBA')).astype(float); rgb=a[...,:3]/255
    lin=s2l(rgb*255); Lm=np.median(lin@[0.2126,0.7152,0.0722])
    L=lin@[0.2126,0.7152,0.0722]; k=gain*np.power(np.clip(L/Lm,1e-4,None),-0.08)
    out=lin*k[...,None]; out=np.where(out<=0.0031308,12.92*out,1.055*np.clip(out,0,None)**(1/2.4)-0.055)
    hsv=mc.rgb_to_hsv(np.clip(out,0,1)); hsv[...,0]=(hsv[...,0]+shift/360)%1; hsv[...,1]=np.clip(hsv[...,1]*sat,0,1)
    a[...,:3]=np.minimum(mc.hsv_to_rgb(hsv)*255,cap); return a
VS=1.0
AMB=0.70; GAIN=1.0; VP=0.6
LDIR=np.array([3,5,4.])/np.linalg.norm([3,5,4])
ROSE=[((252,232,214),.30),((250,212,196),.20),((246,178,176),.20),((236,124,140),.12),((252,214,188),.14),((255,244,232),.04)]
def run(dst,LO,HI,EMI,LG,seed=5):
    rng=np.random.default_rng(seed); g=G('FlowerBank_1m_sample_A.glb'); j=g.j; new={}
    src_of=lambda t: j['textures'][t]['source']
    imgs=lambda i: Image.open(io.BytesIO(g.imgbytes(i)))
    for t in (4,5,6):
        i=src_of(t); im=imgs(i); new[i]=enc(petal_tex(im,LO,HI),j['images'][i]['mimeType'],'A' in im.mode or im.mode=='P')
    for t in (3,7,9):
        i=src_of(t); im=imgs(i); new[i]=enc(leaf_tex(im,LG,-30,0.72),j['images'][i]['mimeType'],'A' in im.mode or im.mode=='P')
    prims=j['meshes'][0]['primitives']; name=lambda p: j['materials'][p['material']]['name']; P={name(p):p for p in prims}
    # 송이별 색(정점색), 원본 정점색 밝기 결은 약하게 남김
    X=np.vstack([g.get(P[n]['attributes']['POSITION']) for n in ('Flowers','PetalEdge')]); cen,_=kmeans2(X[::5],36,minit='++',seed=5,iter=40)
    w=np.array([r[1] for r in ROSE]); pal=np.array([s2l(ROSE[rng.choice(len(ROSE),p=w/w.sum())][0]) for _ in range(36)])
    for n in ('Flowers','PetalEdge'):
        p=P[n]; Xp=g.get(p['attributes']['POSITION']); C=g.get(p['attributes']['COLOR_0'])
        l=np.argmin(((Xp[:,None]-cen[None])**2).sum(-1),1); lum=C[:,:3]@[0.2126,0.7152,0.0722]; v=np.ones(len(Xp))
        for b in np.unique(l):
            m=l==b; v[m]=np.clip(lum[m]/np.percentile(lum[m],90),0.25,1)**VP
        Nn=g.get(p['attributes']['NORMAL']); ndl=np.abs(Nn@LDIR)
        shade=(AMB+(1-AMB)*ndl)*v
        C[:,:3]=np.clip(np.power(pal[l],VS)*shade[:,None]*GAIN*np.array([1.06,1.0,0.93]),0,1); g.put(p['attributes']['COLOR_0'],C)
    for nm in ('Leaf','LeafCard','Base'):
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); vv=C[:,:3].mean(1); vmax=np.percentile(vv,98)
        C[:,:3]=(0.6+0.4*np.clip(vv/vmax,0,1))[:,None]; g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        if m['name'] in ('Flowers','PetalEdge'):
            m.pop('emissiveFactor',None); m.pop('emissiveTexture',None); m['extensions']={'KHR_materials_unlit':{}}   # 빛은 정점색에 구움(텍스처 결은 그대로)
        if m['name']=='Leaf': m['emissiveTexture']={'index':7}; m['emissiveFactor']=[0.10,0.10,0.07]
        if m['name']=='LeafCard': m['emissiveFactor']=[0.08,0.08,0.06]
    j['extensionsUsed']=['KHR_materials_unlit']
    g.save(dst,new)
if __name__=='__main__':
    a=list(map(float,sys.argv[1:5])); MIXO=float(sys.argv[5]); VS=float(sys.argv[6]); GAIN=float(sys.argv[7]) if len(sys.argv)>7 else 1.0; run('web/_s18a.glb',*a)
