import io,sys,json,numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from fbio import G
def lin(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
ROSE=[('white',(240,238,234),.24),('white',(239,237,232),.14),('ivory',(242,234,214),.22),('cream',(243,231,206),.08),
      ('peach',(245,213,190),.18),('blush',(242,221,212),.08),('pink',(243,208,212),.06)]
HYD=[('white',(238,239,232),.35),('white',(240,238,233),.25),('paleblue',(214,224,240),.15),('blush',(240,222,228),.15),('lime',(226,236,204),.10)]
REPORT={}
def lift(L,ratio=2.2):
    """어두운 쪽만 올림: 중앙값 아래만 감마 곡선, 하위 5%가 ratio배 되도록. L 0~1"""
    M=np.median(L); p5=max(np.percentile(L,5),1e-3); tgt=min(p5*ratio,M*0.95)
    gam=np.log(tgt/M)/np.log(p5/M) if p5<M else 1
    out=np.where(L<M,M*np.power(np.clip(L/M,1e-6,1),gam),L); return out
def tex_lift(img,desat=False,ratio=2.2,name=''):
    a=np.asarray(img.convert('RGBA')).astype(float)/255; rgb=a[...,:3]; m=a[...,3]>0.04
    L=rgb@np.array([0.299,0.587,0.114])
    if desat:
        # 아틀라스 2x2 칸(장미 종류)마다 따로 정규화 → 빨강 칸이 회색 장미로 안 되게
        h,w=L.shape; Ln=np.zeros_like(L)
        for gy in range(2):
            for gx in range(2):
                sl=(slice(gy*h//2,(gy+1)*h//2),slice(gx*w//2,(gx+1)*w//2)); mm=m[sl]
                q=np.percentile(L[sl][mm],97) if mm.any() else 1
                Ln[sl]=np.clip(L[sl]/q,0,1)
        rgb=np.repeat(Ln[...,None],3,-1)*0.94; L=rgb[...,0]
    Lt=L.copy(); Lt[m]=lift(L[m],ratio)
    g=np.where(L>1e-4,Lt/np.maximum(L,1e-4),1)[...,None]; out=np.clip(rgb*g,0,240/255)
    a[...,:3]=out; L2=out@np.array([0.299,0.587,0.114])
    REPORT[name]=dict(before=np.percentile(np.asarray(img.convert('RGBA'))[...,:3][m]@[0.299,0.587,0.114],[5,50,95]).round(0).tolist(),after=np.percentile(L2[m]*255,[5,50,95]).round(0).tolist())
    return a
def enc(a,src_mime,has_alpha):
    buf=io.BytesIO()
    if src_mime.endswith('jpeg') and not has_alpha:
        Image.fromarray((a[...,:3]*255).round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=92); return buf.getvalue(),'image/jpeg'
    Image.fromarray((a*255).round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); return buf.getvalue(),'image/png'
def run(src,dst,K=1.0,seed=5):
    rng=np.random.default_rng(seed); g=G(src); j=g.j
    # 1) sheen 제거 (_1002판 조건)
    for m in j['materials']:
        if 'extensions' in m: m['extensions'].pop('KHR_materials_sheen',None); 
        if m.get('extensions')=={}: m.pop('extensions')
    j['extensionsUsed']=[e for e in j.get('extensionsUsed',[]) if e!='KHR_materials_sheen']
    # 2) 텍스처
    src_of=lambda t: j['textures'][t]['source']
    newimgs={}
    imgs={i:Image.open(io.BytesIO(g.imgbytes(i))) for i in range(len(j['images']))}
    ao_i=src_of(2)  # occlusionTexture index 2
    a=np.asarray(imgs[ao_i].convert('RGB')).astype(float)/255; ao2=np.minimum(1-(1-a)*0.45,240/255)
    REPORT['AO맵']=dict(before=np.percentile(a*255,[5,50,95]).round(0).tolist(),after=np.percentile(ao2*255,[5,50,95]).round(0).tolist())
    newimgs[ao_i]=enc(np.dstack([ao2,np.ones(a.shape[:2])]),'image/png',False)
    for t,desat,nm in [(4,True,'장미 아틀라스'),(5,True,'장미 발광'),(6,True,'꽃잎 가장자리'),(3,False,'잎 카드'),(7,False,'잎'),(9,False,'바닥')]:
        i=src_of(t); im=imgs[i]; ha='A' in im.mode or im.mode=='P'
        newimgs[i]=enc(tex_lift(im,desat,name=nm),j['images'][i]['mimeType'],ha)
    # 3) 정점색
    prims=j['meshes'][0]['primitives']; mat=lambda p: j['materials'][p['material']]['name']
    P={mat(p):p for p in prims}
    pos=lambda p: g.get(p['attributes']['POSITION']); colr=lambda p: g.get(p['attributes']['COLOR_0'])
    # 장미: Flowers+PetalEdge 함께 송이 나눔
    rp=[P['Flowers'],P['PetalEdge']]; allpos=np.vstack([pos(p) for p in rp])
    k=36; cen,_=kmeans2(allpos[::5],k,minit='++',seed=seed,iter=40)
    tg=[]
    for b in range(k):
        w=np.array([q[2] for q in ROSE]); q=ROSE[rng.choice(len(ROSE),p=w/w.sum())]
        c=np.array(q[1],float)*rng.uniform(0.975,1)+rng.normal(0,1,3)
        if q[0]=='white': c[2]=np.clip(c[2],c[0]-8,c[0]-4)
        tg.append(lin(np.clip(c,0,240)))
    tg=np.array(tg)
    vstats=[]
    for p in rp:
        X=pos(p); C=colr(p); l=np.argmin(((X[:,None,:]-cen[None])**2).sum(-1),1)
        # 장미 원본 정점색은 송이별 색(빨강·분홍) 차이가 커서 그늘로 쓰면 회색 송이가 생김 → 그늘은 텍스처·AO맵·조명에 맡김
        s=np.ones(len(X))
        C[:,:3]=np.clip(tg[l]*K*(0.45+0.55*s)[:,None],0,1); g.put(p['attributes']['COLOR_0'],C)
    # 수국: 한 송이 안을 작은 무리로 나눠 흰·연한 색 섞기
    p=P['Hydrangea_1']; X=pos(p); C=colr(p); cen2,l=kmeans2(X,8,minit='++',seed=seed)
    lum=C[:,:3]@[0.2126,0.7152,0.0722]; s=np.clip(lum/np.percentile(lum,80),0,1)
    hc=[]
    for b in range(8):
        w=np.array([q[2] for q in HYD]); q=HYD[rng.choice(len(HYD),p=w/w.sum())]; hc.append(lin(np.array(q[1],float)))
    C[:,:3]=np.clip(np.array(hc)[l]*K*(0.45+0.55*s)[:,None],0,1); g.put(p['attributes']['COLOR_0'],C)
    p=P['HydrangeaCore']; C=colr(p); C[:,:3]=lin((200,210,190)); g.put(p['attributes']['COLOR_0'],C)
    # 잎·바닥 정점색(구운 그늘): 어두운 쪽만 올림
    for nm in ['Leaf','LeafCard','Base']:
        p=P[nm]; C=colr(p); L=C[:,:3].mean(1); Lt=lift(L)
        REPORT['정점그늘 '+nm]=dict(before=(np.percentile(L,[5,50,95])*255).round(0).tolist(),after=(np.percentile(Lt,[5,50,95])*255).round(0).tolist())
        C[:,:3]=np.clip(C[:,:3]*(Lt/np.maximum(L,1e-4))[:,None],0,1); g.put(p['attributes']['COLOR_0'],C)
    g.save(dst,newimgs)
    print(json.dumps(REPORT,ensure_ascii=False))
if __name__=='__main__':
    run('FlowerBank_1m_sample_A.glb','web/FlowerBank_1m_sample_A_1002.glb',K=0.0+float(sys.argv[1]) if len(sys.argv)>1 else 1.0)
