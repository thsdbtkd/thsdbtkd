"""샘플 꽃둑 20판: Blender(Cycles)로 만든 작약(겹꽃잎 104장·꽃잎 텍스처·구운 빛)으로 원본 꽃 교체.
잎·바닥은 원본 재질 유지 + 19판 잎 보정(밝기·노란 기). 흰 잔꽃·라벤더 + 날리는 꽃잎."""
import sys,io,numpy as np
from scipy.cluster.vq import kmeans2
from PIL import Image
from fbio import G
from peony_glb import peony_arrays, tex_bytes
import sampleA_v19 as V19
from sampleA_v15 import add_prim
from recolor11 import floret,sprig
from recolor5 import s2l
PAL=[((255,240,228),(248,196,176),.26),((254,222,208),(244,160,150),.24),((252,198,190),(238,128,134),.20),
     ((248,158,160),(228,96,114),.10),((255,228,204),(246,176,146),.16),((255,248,240),(252,218,204),.04)]
def run(dst,SIZE=1.25,NF=520,NL=220,seed=21):
    rng=np.random.default_rng(seed)
    # 1) 잎 보정만 19판 방식으로 (꽃은 버림)
    V19.FEM=0.12; V19.run('web/_s20base.glb',3.5,0.07,3.4,0.10,0.48)
    g=G('web/_s20base.glb'); j=g.j; prims=j['meshes'][0]['primitives']; name=lambda p: j['materials'][p['material']]['name']; P={name(p):p for p in prims}
    X=np.vstack([g.get(P[n]['attributes']['POSITION']) for n in ('Flowers','PetalEdge')])
    cen,lab=kmeans2(X,36,minit='++',seed=5,iter=40)
    LX=g.get(P['Leaf']['attributes']['POSITION'])
    hp=g.get(P['Hydrangea_1']['attributes']['POSITION']); hc=hp.mean(0); hr=np.ptp(hp,0)
    core=np.array([X[:,0].mean(),X[:,1].min()-0.05,X[:,2].mean()])
    blooms=[]
    for b in range(36):
        m=lab==b
        if m.sum()<80: continue
        R=0.5*np.sort(np.ptp(X[m],0))[-2]*SIZE
        ax=cen[b]-core; ax[1]=max(ax[1],0.05); ax/=np.linalg.norm(ax); ax=0.45*ax+0.55*np.array([0,1,0.]); ax/=np.linalg.norm(ax)
        blooms.append([cen[b].copy(),ax,R])
    Rm=np.median([b[2] for b in blooms])
    for dx,dz in [(-0.18,-0.12),(0.2,0.05),(-0.1,0.28)]:
        blooms.append([hc+np.array([dx*hr[0],0,dz*hr[2]]),np.array([0,1,0.]),Rm*rng.uniform(0.95,1.08)])
    w=np.array([p[2] for p in PAL]); items=[]
    for c,ax,R in blooms:
        d2=((LX[:,[0,2]]-c[[0,2]])**2).sum(1); near=LX[d2<(R*1.1)**2]
        if len(near): c[1]=min(c[1],np.percentile(near[:,1],80)+R*0.05)
        k=rng.choice(len(PAL),p=w/w.sum()); base,deep,_=PAL[k]
        items.append((int(rng.integers(4)),c-ax*R*0.15,ax,R,rng.uniform(0,6.28),base,deep))
    Pp,Np,Up,Cp,Tp=peony_arrays(items,rng)
    j['meshes'][0]['primitives']=[p for p in prims if name(p) not in ('Flowers','PetalEdge','Hydrangea_1','HydrangeaCore')]
    # 작약 primitive 추가(텍스처 이미지 하나 더)
    def addv(arr,target=34962):
        while len(g.bin)%4: g.bin.append(0)
        off=len(g.bin); g.bin+=arr.tobytes(); j['bufferViews'].append({'buffer':0,'byteOffset':off,'byteLength':len(arr.tobytes()) if not isinstance(arr,bytes) else len(arr),'target':target} if target else {'buffer':0,'byteOffset':off,'byteLength':len(arr)}); return len(j['bufferViews'])-1
    def acc(arr,typ,ct=5126):
        a={'bufferView':addv(np.ascontiguousarray(arr)),'componentType':ct,'count':len(arr),'type':typ}
        if typ=='VEC3' and ct==5126 and arr is Pp: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
        j['accessors'].append(a); return len(j['accessors'])-1
    ip=acc(Pp,'VEC3'); inn=acc(Np,'VEC3'); iu=acc(Up,'VEC2'); ic=acc(np.hstack([Cp,np.ones((len(Cp),1),np.float32)]),'VEC4')
    T=Tp.ravel().astype(np.uint32); bvi=addv(T,34963); j['accessors'].append({'bufferView':bvi,'componentType':5125,'count':len(T),'type':'SCALAR'}); ii=len(j['accessors'])-1
    png=tex_bytes()
    while len(g.bin)%4: g.bin.append(0)
    off=len(g.bin); g.bin+=png; j['bufferViews'].append({'buffer':0,'byteOffset':off,'byteLength':len(png)})
    j['images'].append({'bufferView':len(j['bufferViews'])-1,'mimeType':'image/png'}); j['textures'].append({'source':len(j['images'])-1,'sampler':0})
    j['materials'].append({'name':'Peony','pbrMetallicRoughness':{'baseColorTexture':{'index':len(j['textures'])-1},'metallicFactor':0,'roughnessFactor':0.65},'alphaMode':'MASK','alphaCutoff':0.5,'doubleSided':True,'extensions':{'KHR_materials_unlit':{}}})
    j.setdefault('extensionsUsed',[]); 
    if 'KHR_materials_unlit' not in j['extensionsUsed']: j['extensionsUsed'].append('KHR_materials_unlit')
    j['meshes'][0]['primitives'].append({'attributes':{'POSITION':ip,'NORMAL':inn,'TEXCOORD_0':iu,'COLOR_0':ic},'indices':ii,'material':len(j['materials'])-1})
    j['buffers'][0]['byteLength']=len(g.bin)
    # 잔꽃
    lp=P['Leaf']; Xl=LX; Nl=g.get(lp['attributes']['NORMAL']); t=g.get(lp['indices']).astype(np.int64).reshape(-1,3)
    tn=Nl[t].mean(1); tn/=np.linalg.norm(tn,axis=1,keepdims=True)+1e-9; tc=Xl[t].mean(1)
    ar=np.linalg.norm(np.cross(Xl[t[:,1]]-Xl[t[:,0]],Xl[t[:,2]]-Xl[t[:,0]]),axis=1)/2
    ok=(tn[:,1]>0.25)&(tc[:,1]>np.percentile(Xl[:,1],35)); pr=ar*ok; pr/=pr.sum(); r=0.0085; PP=[];CC=[];TT=[];base=0
    for k,ti in enumerate(rng.choice(len(t),NF+NL,p=pr)):
        u,v=rng.random(2)
        if u+v>1: u,v=1-u,1-v
        q=Xl[t[ti,0]]+u*(Xl[t[ti,1]]-Xl[t[ti,0]])+v*(Xl[t[ti,2]]-Xl[t[ti,0]]); nn=tn[ti]; q=q+nn*r*rng.uniform(0.8,1.8)
        if k<NF: pp,cc,tt=floret(q,nn,r*rng.uniform(0.75,1.3),rng,np.array([255,250,240.]),np.array([240,206,110.]))
        else: pp,cc,tt=sprig(q,nn,r*0.85,rng,np.array([206,172,228.]))
        PP.append(pp);CC.append(cc);TT.append(tt+base); base+=len(pp)
    add_prim(g,np.vstack(PP),s2l(np.clip(np.vstack(CC),0,252)),np.vstack(TT),'FillerFlowers')
    g.save(dst,{}); print('작약',len(items),'작약 삼각형',len(T)//3)
if __name__=='__main__': run('web/FlowerBank_1m_sample_A_1002_v20.glb',float(sys.argv[1]))
