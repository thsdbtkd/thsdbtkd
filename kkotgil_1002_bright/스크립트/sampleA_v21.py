"""꽃둑 시험판 21: /map/slam 조건용. 원본 샘플 잎·바닥(조명 받는 재질) + Blender 작약·장미(얇고 밝은 꽃잎, 림라이트·햇살 구움, 조명 없는 재질)
+ 흰 잔꽃(옆·아래로 흘러내리게)·라벤더. 꽃잎 날림은 뷰어 내장 효과 사용(GLB 없음). 삼각형 7만·10MB 이하."""
import os,sys,io,numpy as np
os.environ.setdefault('PEONY_DIR','p21/')
from scipy.cluster.vq import kmeans2
from PIL import Image
from fbio import G
import peony_glb as PG
from sampleA_v15 import add_prim
from recolor11 import floret,sprig
from recolor5 import s2l
import sampleA_v19 as V19
PAL=[((255,232,214),(250,190,160),.26),((254,214,200),(246,150,140),.26),((252,192,188),(240,122,132),.20),
     ((248,156,160),(232,96,116),.10),((255,222,196),(250,170,138),.18)]
def leaf_grade(img):
    return V19.grade(img,3.0,0.08,(1.24,1.08,0.60),0.78)   # 노란 연두·밝게
def run(dst,SIZE,NF,NL,seed=21):
    rng=np.random.default_rng(seed); g=G('FlowerBank_1m_sample_A.glb'); j=g.j; new={}
    for t in (3,7,9):
        i=j['textures'][t]['source']; im=Image.open(io.BytesIO(g.imgbytes(i)))
        a=leaf_grade(im); new[i]=V19.enc(a,'image/jpeg' if not ('A' in im.mode or im.mode=='P') else 'image/png','A' in im.mode or im.mode=='P')
    prims=j['meshes'][0]['primitives']; name=lambda p: j['materials'][p['material']]['name']; P={name(p):p for p in prims}
    for nm in ('Leaf','LeafCard','Base'):
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); C[:,:3]=np.clip(C[:,:3]+(1-C[:,:3])*0.45,0,1); g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        if 'occlusionTexture' in m: m['occlusionTexture']['strength']=0.3
        if m['name']=='Leaf': m['emissiveTexture']={'index':7}; m['emissiveFactor']=[0.14,0.13,0.07]
    X=np.vstack([g.get(P[n]['attributes']['POSITION']) for n in ('Flowers','PetalEdge')]); cen,lab=kmeans2(X,36,minit='++',seed=5,iter=40)
    LX=g.get(P['Leaf']['attributes']['POSITION'])
    hp=g.get(P['Hydrangea_1']['attributes']['POSITION']); hc=hp.mean(0); hr=np.ptp(hp,0)
    core=np.array([X[:,0].mean(),X[:,1].min()-0.05,X[:,2].mean()]); blooms=[]
    for b in range(36):
        m=lab==b
        if m.sum()<80: continue
        R=0.5*np.sort(np.ptp(X[m],0))[-2]*SIZE
        ax=cen[b]-core; ax[1]=max(ax[1],0.05); ax/=np.linalg.norm(ax); ax=0.45*ax+0.55*np.array([0,1,0.]); ax/=np.linalg.norm(ax)
        blooms.append([cen[b].copy(),ax,R])
    Rm=np.median([b[2] for b in blooms])
    for dx,dz in [(-0.15,-0.1),(0.18,0.15)]:
        blooms.append([hc+np.array([dx*hr[0],0,dz*hr[2]]),np.array([0,1,0.]),Rm])
    w=np.array([p[2] for p in PAL]); items=[]
    for c,ax,R in blooms:
        d2=((LX[:,[0,2]]-c[[0,2]])**2).sum(1); near=LX[d2<(R*1.1)**2]
        if len(near): c[1]=min(c[1],np.percentile(near[:,1],80)+R*0.05)
        k=rng.choice(len(PAL),p=w/w.sum()); base,deep,_=PAL[k]
        rose=rng.random()<0.25; si=int(rng.integers(4,6)) if rose else int(rng.integers(0,4))
        items.append((si,c-ax*R*0.12,ax,R*(0.8 if rose else 1.0),rng.uniform(0,6.28),base,deep))
    Pp,Np,Up,Cp,Tp=PG.peony_arrays(items,rng)
    j['meshes'][0]['primitives']=[p for p in prims if name(p) not in ('Flowers','PetalEdge','Hydrangea_1','HydrangeaCore')]
    def addv(b,target=None):
        while len(g.bin)%4: g.bin.append(0)
        off=len(g.bin); g.bin+=b; bv={'buffer':0,'byteOffset':off,'byteLength':len(b)}
        if target: bv['target']=target
        j['bufferViews'].append(bv); return len(j['bufferViews'])-1
    def acc(arr,typ,ct=5126,minmax=False,target=34962):
        a={'bufferView':addv(np.ascontiguousarray(arr).tobytes(),target),'componentType':ct,'count':len(arr),'type':typ}
        if minmax: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
        j['accessors'].append(a); return len(j['accessors'])-1
    ip=acc(Pp,'VEC3',minmax=True); inn=acc(Np,'VEC3'); iu=acc(Up,'VEC2'); ic=acc(np.hstack([Cp,np.ones((len(Cp),1),np.float32)]).astype(np.float32),'VEC4')
    ii=acc(Tp.ravel().astype(np.uint32),'SCALAR',5125,target=34963)
    png=PG.tex_bytes(); bi=addv(png); j['images'].append({'bufferView':bi,'mimeType':'image/png'}); j['textures'].append({'source':len(j['images'])-1,'sampler':0})
    j['materials'].append({'name':'Peony','pbrMetallicRoughness':{'baseColorTexture':{'index':len(j['textures'])-1},'metallicFactor':0,'roughnessFactor':0.65},'alphaMode':'MASK','alphaCutoff':0.5,'doubleSided':True,'extensions':{'KHR_materials_unlit':{}}})
    eu=j.setdefault('extensionsUsed',[]); 
    for e in ('KHR_materials_unlit',):
        if e not in eu: eu.append(e)
    j['meshes'][0]['primitives'].append({'attributes':{'POSITION':ip,'NORMAL':inn,'TEXCOORD_0':iu,'COLOR_0':ic},'indices':ii,'material':len(j['materials'])-1})
    j['buffers'][0]['byteLength']=len(g.bin)
    # 흰 잔꽃: 윗면 + 옆면(흘러내리게, 아래로 갈수록 많이)
    lp=P['Leaf']; Nl=g.get(lp['attributes']['NORMAL']); t=g.get(lp['indices']).astype(np.int64).reshape(-1,3)
    tn=Nl[t].mean(1); tn/=np.linalg.norm(tn,axis=1,keepdims=True)+1e-9; tc=LX[t].mean(1)
    ar=np.linalg.norm(np.cross(LX[t[:,1]]-LX[t[:,0]],LX[t[:,2]]-LX[t[:,0]]),axis=1)/2
    side=(np.abs(tn[:,1])<0.55)&(tc[:,1]>LX[:,1].min()+0.02)
    top=(tn[:,1]>0.25)&(tc[:,1]>np.percentile(LX[:,1],35))
    pr=ar*(top*1.0+side*1.6); pr/=pr.sum(); r=0.0095; PP=[];CC=[];TT=[];base=0
    for k,ti in enumerate(rng.choice(len(t),NF+NL,p=pr)):
        u,v=rng.random(2)
        if u+v>1: u,v=1-u,1-v
        q=LX[t[ti,0]]+u*(LX[t[ti,1]]-LX[t[ti,0]])+v*(LX[t[ti,2]]-LX[t[ti,0]]); nn=tn[ti]; q=q+nn*r*rng.uniform(0.8,1.6)
        lit=0.9+0.1*max(0,nn[1])
        if k<NF: pp,cc,tt=floret(q,nn,r*rng.uniform(0.8,1.3),rng,np.array([255,252,244.])*lit,np.array([244,214,120.])*lit)
        else: pp,cc,tt=sprig(q,nn,r*0.85,rng,np.array([210,178,232.])*lit)
        PP.append(pp);CC.append(cc);TT.append(tt+base); base+=len(pp)
    add_prim(g,np.vstack(PP),s2l(np.clip(np.vstack(CC),0,252)),np.vstack(TT),'FillerFlowers')
    g.save(dst,new); print('송이',len(items),'꽃 삼각형',len(Tp))
if __name__=='__main__':
    run('web/FlowerBank_1m_sample_A_1002_v21.glb',float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
