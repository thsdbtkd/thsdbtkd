import io,sys,json,numpy as np,colorsys
from PIL import Image
from fbio import G
def s2l(c): c=np.asarray(c,float); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)
def hue_shift(rgb,deg,sat=1.0):
    # rgb 0~1 sRGB, 색상만 deg 만큼 이동(음수 = 노랑 쪽 = 연두)
    import matplotlib.colors as mc
    hsv=mc.rgb_to_hsv(np.clip(rgb,0,1)); hsv[...,0]=(hsv[...,0]+deg/360)%1; hsv[...,1]=np.clip(hsv[...,1]*sat,0,1); return mc.hsv_to_rgb(hsv)
REP={}
SAT=1.10
def leaf_tex(img,gain,gam,shift=-10,cap=220,name=''):
    a=np.asarray(img.convert('RGBA')).astype(float)/255; rgb=a[...,:3]; m=a[...,3]>0.04
    b0=(rgb[m]*255).mean(1)
    lin=s2l(rgb); L=lin@[0.2126,0.7152,0.0722]; Lm=np.median(L[m])
    # 곡선: 밝기 비율(결)은 감마로 남기고 전체를 gain 배
    k=gain*np.power(np.clip(L/Lm,1e-4,None),gam-1)
    out=l2s(lin*k[...,None]); out=hue_shift(out,shift,SAT); out=np.minimum(out,cap/255)
    a[...,:3]=out; b1=(out[m]*255).mean(1)
    REP[name]=dict(before=np.percentile(b0,[5,50,95]).round(0).tolist(),after=np.percentile(b1,[5,50,95]).round(0).tolist())
    return a
def enc(a,mime,alpha):
    buf=io.BytesIO()
    if mime.endswith('jpeg') and not alpha:
        Image.fromarray((a[...,:3]*255).round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=92); return buf.getvalue(),'image/jpeg'
    Image.fromarray((a*255).round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); return buf.getvalue(),'image/png'
def run(src,dst,gain,gam,vfloor=0.55):
    g=G(src); j=g.j; newimgs={}
    src_of=lambda t: j['textures'][t]['source']
    for t,nm in [(3,'LeafCard 텍스처'),(7,'Leaf 텍스처'),(9,'Base 텍스처')]:
        i=src_of(t); im=Image.open(io.BytesIO(g.imgbytes(i))); al='A' in im.mode or im.mode=='P'
        newimgs[i]=enc(leaf_tex(im,gain,gam,name=nm),j['images'][i]['mimeType'],al)
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    for nm in ['Leaf','LeafCard','Base']:
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); v=C[:,:3].mean(1); vmax=np.percentile(v,98)
        # 구운 그늘: 최저 계수 vfloor, 상대 명암은 남김
        nv=vfloor+(1-vfloor)*np.clip(v/vmax,0,1)
        REP['정점그늘 '+nm]=dict(before=np.percentile(v,[0,5,50]).round(2).tolist(),after=np.percentile(nv,[0,5,50]).round(2).tolist())
        C[:,:3]=nv[:,None]; g.put(p['attributes']['COLOR_0'],C)
    g.save(dst,newimgs); print(json.dumps(REP,ensure_ascii=False))
if __name__=='__main__':
    run('web/FlowerBank_1m_sample_A_1002.glb','web/FlowerBank_1m_sample_A_1002_v4.glb',float(sys.argv[1]),float(sys.argv[2]))
