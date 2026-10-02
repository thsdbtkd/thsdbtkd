import io,sys,numpy as np
from PIL import Image
from fbio import G
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
WARM=np.array([11,4,-11.])
GAP=np.array([110,150,80.])
def warm_tex(img,gap=False):
    a=np.asarray(img.convert('RGBA')).astype(float); m=a[...,3]>10; rgb=a[...,:3]
    rgb=rgb+WARM                                         # 따뜻한 햇살(텍스처)
    if gap:                                              # 틈 밝히기: 잎 텍스처 밝기 하위 10% → 따뜻한 녹색 쪽
        br=rgb.mean(-1); p10=np.percentile(br[m],10); w=np.clip((p10-br)/max(p10-np.percentile(br[m],1),1),0,1)*0.8
        w=np.where(br<p10,np.maximum(w,0.3),0)
        rgb=rgb*(1-w[...,None])+GAP*w[...,None]
        g=rgb.mean(-1,keepdims=True); rgb=g+(rgb-g)*0.80   # 잎이 형광 연두로 뜨지 않게 채도 20% 낮춤
    a[...,:3]=np.clip(rgb,0,245); return a
def enc(a,mime,alpha):
    buf=io.BytesIO()
    if mime.endswith('jpeg') and not alpha:
        Image.fromarray(a[...,:3].round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=92); return buf.getvalue(),'image/jpeg'
    Image.fromarray(a.round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); return buf.getvalue(),'image/png'
def run(src,dst,CREAM,LEAFE,RS):
    g=G(src); j=g.j; src_of=lambda t: j['textures'][t]['source']; new={}
    for t,gap in [(0,False),(4,False),(6,False),(3,True),(7,True),(9,True)]:
        i=src_of(t); im=Image.open(io.BytesIO(g.imgbytes(i))); al='A' in im.mode or im.mode=='P'
        new[i]=enc(warm_tex(im,gap),j['images'][i]['mimeType'],al)
    cream=s2l((245,232,200))
    for m in j['materials']:
        n=m['name']
        if n in ('Flowers','PetalEdge'):                 # 크림색 따뜻한 발광 더하기(+CREAM)
            e=np.array(m.get('emissiveFactor',[0,0,0]))*np.array([1.03,1.0,0.90])+CREAM*cream/cream.max()
            m['emissiveFactor']=e.round(4).tolist()
        if n=='Leaf':  m['emissiveTexture']={'index':7}; m['emissiveFactor']=[LEAFE]*3
        if n=='Base':  m['emissiveTexture']={'index':9}; m['emissiveFactor']=[LEAFE]*3
        if n=='LeafCard': m['emissiveFactor']=[max(0.12,LEAFE)]*3
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    for nm in ('Flowers','PetalEdge'):
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); C[:,:3]*=RS; g.put(p['attributes']['COLOR_0'],C)
    g.save(dst,new)
if __name__=='__main__':
    CREAM,LEAFE,RS=map(float,sys.argv[1:4])
    run('web/FlowerBank_1m_sample_A_1002_v6.glb','web/FlowerBank_1m_sample_A_1002_v7.glb',CREAM,LEAFE,RS)
