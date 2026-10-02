import io,json,numpy as np
from PIL import Image
from glbio import load,save
from recolor5 import s2l,l2s
from fbio import G
R='/home/user/thsdbtkd/kkotgil_1002_bright/glb/'
def recolor(src,dst):
    d=load(src); c=l2s(d['col']).astype(float); R_,G_,B_=c[:,0],c[:,1],c[:,2]
    g=((G_>R_+10)&(G_>=B_))|((G_>=R_-12)&(np.minimum(R_,G_)>B_+50))
    out=c.copy()
    L=c[g].mean(1); r=L/136.4; r2=np.where(r<=1,r,np.minimum(1+0.4*(r-1),1.18))
    out[g]=np.array([124,160,92.])[None]*r2[:,None]
    ng=~g; L2=c[ng].mean(1); k=np.where(L2<150,(L2+(150-L2)*0.45)/np.maximum(L2,1e-6),1.0); out[ng]=c[ng]*k[:,None]
    mx=out.max(1); sat=(mx-out.min(1))/np.maximum(mx,1e-6)
    wf=ng&(sat<0.10); Lw=out[wf].mean(1)*1.04; out[wf]=Lw[:,None]*np.array([1.015,1.0,0.965])[None]/0.9933
    mx=out.max(1); sat=(mx-out.min(1))/np.maximum(mx,1e-6)
    pe=ng&(out[:,0]>out[:,1]+12)&(out[:,1]>out[:,2])&(sat>0.15); p=out[pe]*1.06; out[pe]=p+(255-p)*0.07
    out=np.clip(out,0,242); save(d,s2l(out),dst)
    print(dst.split('/')[-1],'잎',g.sum(),'흰꽃',wf.sum(),'피치',pe.sum(),'정점',len(c))
def sample(src,dst):
    g=G(src); j=g.j; new={}
    for t in (3,7,9):
        i=j['textures'][t]['source']; im=Image.open(io.BytesIO(g.imgbytes(i))); al='A' in im.mode or im.mode=='P'
        a=np.asarray(im.convert('RGBA')).astype(float); rgb=a[...,:3]
        m=(rgb[...,1]>rgb[...,0]+10)&(rgb[...,1]>=rgb[...,2])
        gray=rgb.mean(-1,keepdims=True); mixed=(rgb*0.7+gray*0.3)*0.93
        rgb[m]=mixed[m]; a[...,:3]=np.clip(rgb,0,255)
        buf=io.BytesIO(); mime=j['images'][i]['mimeType']
        if mime.endswith('jpeg') and not al: Image.fromarray(a[...,:3].round().astype(np.uint8),'RGB').save(buf,'JPEG',quality=92); new[i]=(buf.getvalue(),'image/jpeg')
        else: Image.fromarray(a.round().astype(np.uint8),'RGBA').save(buf,'PNG',optimize=True); new[i]=(buf.getvalue(),'image/png')
        print('샘플 텍스처',t,'초록 화소',int(m.sum()))
    g.save(dst,new)
recolor(R+'vr27n_D_bank_bright_v13A.glb',R+'vr27n_D_bank_bright_v18A.glb')
recolor(R+'mound_c_bright_v13A.glb',R+'mound_c_bright_v18A.glb')
sample(R+'FlowerBank_1m_sample_A_1002_v7.glb',R+'FlowerBank_1m_sample_A_1002_v14.glb')
