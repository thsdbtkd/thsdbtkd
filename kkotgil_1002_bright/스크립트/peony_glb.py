"""Blender 에서 만든 작약(정점·UV·AO·깊이) → glb. 꽃잎 텍스처(알파) + 정점색(색·AO·햇살) . 재질: 조명 받는 PBR(양면, 알파 마스크)"""
import json,struct,io,numpy as np
from PIL import Image
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
import os
D=os.environ.get('PEONY_DIR','peony/')
SHAPES=[{k:np.load(f'{D}peony{i}_{k}.npy') for k in ('v','n','t','uv','ao','dep','light')} for i in range(4)]
LGAM=0.38; FLOOR=0.50
def colorize(sh,base,deep,rng,light=np.array([-0.5,0.4,0.75]),ao_k=0.35):
    """송이 하나 정점색(선형): 깊이 그러데이션(바깥 base → 안쪽 deep) × AO × 햇살"""
    a=sh['ao']; s=np.clip(a/max(np.percentile(a,90),1e-3),0,1)**ao_k
    d=np.clip(sh['dep'],0,1)
    col=s2l(base)[None]*(1-d[:,None]**1.1*0.75)+s2l(deep)[None]*(d[:,None]**1.1*0.75)
    lt=sh['light']; lum=lt@[0.2126,0.7152,0.0722]; k=np.percentile(lum,96)
    lt=np.repeat((lum/k)[:,None],3,1)*np.array([1.04,0.98,0.88])[None]   # 밝기만(색 잡티 제거)+따뜻하게
    lt=np.clip(lt,0,1.15); lt=lt**LGAM                       # Cycles 로 구운 빛(따뜻한 해+하늘+반투명) 정규화
    return np.clip(col*(FLOOR+(1-FLOOR)*lt)*rng.uniform(0.97,1.03),0,1)
def place(sh,center,axis,R,spin):
    a=axis/np.linalg.norm(axis); t=np.cross(a,[0,0,1.]) if abs(a[2])<0.9 else np.cross(a,[1,0,0.]); t/=np.linalg.norm(t); b=np.cross(a,t)
    c,s=np.cos(spin),np.sin(spin); t2=c*t+s*b; b2=-s*t+c*b
    M=np.stack([t2,b2,a],0)
    return center[None]+(sh['v']*R)@M, sh['n']@M
def tex_bytes():
    buf=io.BytesIO(); Image.open(D+'petal.png').save(buf,'PNG',optimize=True); return buf.getvalue()
import json,struct
def peony_arrays(items,rng):
    """items: [(shape_i, center, axis, R, spin, base, deep)] → P,N,U,C,T"""
    PP=[];NN=[];UU=[];CC=[];TT=[];nv=0
    for si,c,ax,R,spin,base,deep in items:
        sh=SHAPES[si]; p,n=place(sh,np.asarray(c,float),np.asarray(ax,float),R,spin)
        PP.append(p);NN.append(n);UU.append(sh['uv']);CC.append(colorize(sh,base,deep,rng));TT.append(sh['t']+nv); nv+=len(p)
    U=np.vstack(UU).astype(np.float32).copy(); U[:,1]=1-U[:,1]
    return np.vstack(PP).astype(np.float32),np.vstack(NN).astype(np.float32),U,np.vstack(CC).astype(np.float32),np.vstack(TT).astype(np.uint32)
def write_glb(path,P,N,U,C,T):
    nv=len(P); C4=np.hstack([C,np.ones((nv,1),np.float32)]); T=T.ravel()
    chunks=[P.tobytes(),N.tobytes(),U.tobytes(),C4.tobytes(),T.tobytes(),tex_bytes()]; bv=[];binb=b''
    for ch in chunks:
        while len(binb)%4: binb+=b'\0'
        bv.append({'buffer':0,'byteOffset':len(binb),'byteLength':len(ch)}); binb+=ch
    while len(binb)%4: binb+=b'\0'
    mat={'name':'Peony','pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':0.65},'alphaMode':'MASK','alphaCutoff':0.5,'doubleSided':True,'extensions':{'KHR_materials_unlit':{}}}
    j={'asset':{'version':'2.0'},'extensionsUsed':['KHR_materials_unlit'],'scenes':[{'nodes':[0]}],'scene':0,'nodes':[{'mesh':0}],
     'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'TEXCOORD_0':2,'COLOR_0':3},'indices':4,'material':0}]}],
     'materials':[mat],'textures':[{'source':0,'sampler':0}],'samplers':[{'magFilter':9729,'minFilter':9987}],'images':[{'bufferView':5,'mimeType':'image/png'}],
     'accessors':[{'bufferView':0,'componentType':5126,'count':nv,'type':'VEC3','min':P.min(0).tolist(),'max':P.max(0).tolist()},
                  {'bufferView':1,'componentType':5126,'count':nv,'type':'VEC3'},{'bufferView':2,'componentType':5126,'count':nv,'type':'VEC2'},
                  {'bufferView':3,'componentType':5126,'count':nv,'type':'VEC4'},{'bufferView':4,'componentType':5125,'count':len(T),'type':'SCALAR'}],
     'bufferViews':bv,'buffers':[{'byteLength':len(binb)}]}
    js=json.dumps(j).encode(); js+=b' '*((4-len(js)%4)%4)
    open(path,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb)
