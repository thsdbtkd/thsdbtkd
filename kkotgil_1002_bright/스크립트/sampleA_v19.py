"""19판: 원본 샘플을 하나도 망가뜨리지 않고(조명 받는 재질·sheen·원본 텍스처 색·정점색 그대로) 사진 보정처럼 밝기·따뜻함만 올림.
꽃: 텍스처·정점색에 노출 + 크림 쪽 살짝(색조·명암·결 유지). 잎: 노출·노란 기. 수국: 연분홍 흰색. + 흰 잔꽃·라벤더."""
import io,sys,numpy as np
from PIL import Image
from fbio import G
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
def enc(a,mime,alpha):
    buf=io.BytesIO()
    if mime.endswith('jpeg') and not alpha:
        Image.fromarray(np.clip(a[...,:3],0,255).round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=93); return buf.getvalue(),'image/jpeg'
    Image.fromarray(np.clip(a,0,255).round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); return buf.getvalue(),'image/png'
def grade(img,exp,lift,warm,sat):
    """사진 보정: 선형 노출 → 어두운 쪽 들기(lift) → 따뜻하게 → 채도"""
    a=np.asarray(img.convert('RGBA')).astype(float); lin=s2l(a[...,:3])
    lin=lin*exp; lin=lin/(1+lin*0.12)                     # 부드러운 상한(하이라이트 날림 방지)
    s=l2s(lin); s=s+(255-s)*lift
    s=s*np.array(warm)[None,None]
    g=s.mean(-1,keepdims=True); s=g+(s-g)*sat
    a[...,:3]=np.clip(s,0,252); return a
FW=(1.04,1.0,0.95); LV=0.62; MIX=0.65
FEM=0.13
ROSE=[((252,226,210),.22),((250,200,190),.24),((246,172,176),.24),((236,120,138),.12),((250,206,180),.14),((255,238,226),.04)]; VW=(1.04,0.99,0.90)
def run(dst,FE,FL,LE,LL,VL,seed=7):
    g=G('FlowerBank_1m_sample_A.glb'); j=g.j; new={}
    src_of=lambda t: j['textures'][t]['source']; imgs=lambda i: Image.open(io.BytesIO(g.imgbytes(i)))
    for t in (4,5,6):
        i=src_of(t); im=imgs(i); new[i]=enc(grade(im,FE,FL,FW,1.0),j['images'][i]['mimeType'],'A' in im.mode or im.mode=='P')
    for t in (3,7,9):
        i=src_of(t); im=imgs(i); new[i]=enc(grade(im,LE,LL,(1.16,1.06,0.70),0.90),j['images'][i]['mimeType'],'A' in im.mode or im.mode=='P')
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    from scipy.cluster.vq import kmeans2
    rng=np.random.default_rng(seed)
    X=np.vstack([g.get(P[n]['attributes']['POSITION']) for n in ('Flowers','PetalEdge')]); cen,_=kmeans2(X[::5],36,minit='++',seed=5,iter=40)
    w=np.array([r[1] for r in ROSE]); pal=np.array([ROSE[rng.choice(len(ROSE),p=w/w.sum())][0] for _ in range(36)],float)
    for n in ('Flowers','PetalEdge'):
        p=P[n]; C=g.get(p['attributes']['COLOR_0']); s=l2s(C[:,:3]); s=s+(255-s)*VL      # 정점색 밝게
        s=s*np.array(VW)
        Xp=g.get(p['attributes']['POSITION']); l=np.argmin(((Xp[:,None]-cen[None])**2).sum(-1),1)
        lum=C[:,:3]@[0.2126,0.7152,0.0722]; v=np.ones(len(l))
        for b in np.unique(l):
            m=l==b; v[m]=np.clip(lum[m]/np.percentile(lum[m],90),0.3,1)**0.45
        tgt=l2s(s2l(pal[l])*v[:,None])
        s=s*(1-MIX)+tgt*MIX                                                       # 송이 색을 목표 사진 색 쪽으로(명암은 원본)
        C[:,:3]=s2l(np.clip(s,0,255)); g.put(p['attributes']['COLOR_0'],C)
    for n in ('Hydrangea_1','HydrangeaCore'):
        p=P[n]; C=g.get(p['attributes']['COLOR_0']); s=l2s(C[:,:3]); L=s.mean(1,keepdims=True)
        s=np.array([250,206,204.])[None]*(0.75+0.25*L/L.max()); C[:,:3]=s2l(np.clip(s,0,255)); g.put(p['attributes']['COLOR_0'],C)
    for nm in ('Leaf','LeafCard','Base'):
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); v=C[:,:3].mean(1); C[:,:3]=np.clip(C[:,:3]+(1-C[:,:3])*LV,0,1); g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        if 'occlusionTexture' in m: m['occlusionTexture']['strength']=0.35
        if m['name'] in ('Flowers','PetalEdge'): m['emissiveFactor']=[FEM*1.03,FEM,FEM*0.95]
        if m['name']=='Hydrangea_1': m['emissiveFactor']=[0.10,0.07,0.07]
    g.save(dst,new)
if __name__=='__main__':
    a=list(map(float,sys.argv[1:6])); FEM=float(sys.argv[6]) if len(sys.argv)>6 else 0.13; run('web/_s19a.glb',*a)
