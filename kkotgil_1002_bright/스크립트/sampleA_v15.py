"""샘플 꽃둑 15판: 고객 「개선된 AR 화면」 사진처럼. 7판(잎·텍스처) 위에
- 장미·수국: 조명 없는 재질 + 정점에 따뜻한 빛 굽기(10판 방식), 팔레트 = 참고 사진 k-평균 색
- 잎: 참고 사진 잎 색(중앙 105,118,72 · 밝은 쪽 117) 쪽
- 흰 잔꽃·라벤더 잔꽃: 잎 위에 새 primitive(조명 없음)로 추가"""
import sys,io,json,numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from fbio import G
from sampleA_v10 import s2l,l2s,bake_rows
import sampleA_v10 as SA
ROSE=[((252,228,224),.16),((250,206,204),.22),((246,172,184),.26),((236,124,148),.20),((250,212,192),.08),((255,246,238),.08)]
HYD=[((250,212,214),.40),((255,242,234),.35),((214,190,228),.25)]   # 수국 자리: 연분홍·크림·라벤더 잔꽃 무리
LDIR=np.array([3,5,4.])/np.linalg.norm([3,5,4])
def pick(rng,pal):
    w=np.array([p[1] for p in pal]); return np.array(pal[rng.choice(len(pal),p=w/w.sum())][0],float)
def add_prim(g,P,C,T,matname):
    j=g.j; P=P.astype(np.float32); C=np.hstack([C,np.ones((len(C),1))]).astype(np.float32); T=T.astype(np.uint32).ravel()
    def addview(arr,target):
        while len(g.bin)%4: g.bin.append(0)
        off=len(g.bin); g.bin+=arr.tobytes(); j['bufferViews'].append({'buffer':0,'byteOffset':off,'byteLength':arr.nbytes,'target':target}); return len(j['bufferViews'])-1
    ip=len(j['accessors']); j['accessors'].append({'bufferView':addview(P,34962),'componentType':5126,'count':len(P),'type':'VEC3','min':P.min(0).tolist(),'max':P.max(0).tolist()})
    ic=len(j['accessors']); j['accessors'].append({'bufferView':addview(C,34962),'componentType':5126,'count':len(C),'type':'VEC4'})
    ii=len(j['accessors']); j['accessors'].append({'bufferView':addview(T,34963),'componentType':5125,'count':len(T),'type':'SCALAR'})
    j['materials'].append({'name':matname,'pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':1},'doubleSided':True,'extensions':{'KHR_materials_unlit':{}}})
    j['meshes'][0]['primitives'].append({'attributes':{'POSITION':ip,'COLOR_0':ic},'indices':ii,'material':len(j['materials'])-1})
    j['buffers'][0]['byteLength']=len(g.bin)
LEMI=0.15
def run(dst,AMB=0.62,LEAFMUL=(0.80,0.70,0.62),NF=420,NL=80,seed=5):
    rng=np.random.default_rng(seed)
    # 장미·수국 빛 굽기: 10판 함수에 팔레트만 바꿔 넣음
    SA.ROSE=ROSE; SA.HYD=HYD; SA.pick=pick; SA.SAT=0.45
    SA.LEAFMUL=np.array(LEAFMUL)
    def bake_rows(T,shade):
        tl=s2l(T); return l2s(np.power(tl,1+0.35*(1-shade[:,None]))*np.power(shade[:,None],0.9))
    SA.bake_rows=bake_rows
    SA.run('web/FlowerBank_1m_sample_A_1002_v7.glb','FlowerBank_1m_sample_A.glb','web/_tmp15.glb',AMB)
    g=G('web/_tmp15.glb'); j=g.j
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    # 따뜻한 햇살 색(꽃 정점): R+, B−
    for n in ('Flowers','PetalEdge','Hydrangea_1'):
        p=P[n]; C=g.get(p['attributes']['COLOR_0']); C[:,0]*=0.95; C[:,2]*=1.12;   # 7판 텍스처에 들어간 따뜻한 기운(R+11·B−11) 상쇄 C[:,:3]=np.clip(C[:,:3],0,1); g.put(p['attributes']['COLOR_0'],C)
    # 잎 발광(7판 0.35)은 잎을 형광 연두로 만듦 → 줄이고 노란 기(R>G 쪽)로
    for m in j['materials']:
        if m['name'] in ('Leaf','Base'): m['emissiveFactor']=[LEMI*1.05,LEMI*0.95,LEMI*0.8]
        if m['name']=='LeafCard': m['emissiveFactor']=[0.10,0.09,0.06]
    # 잔꽃 심기: 위·바깥을 향한 잎 면에
    lp=P['Leaf']; X=g.get(lp['attributes']['POSITION']); N=g.get(lp['attributes']['NORMAL']); t=g.get(lp['indices']).astype(np.int64).reshape(-1,3)
    tn=N[t].mean(1); tn/=np.linalg.norm(tn,axis=1,keepdims=True)+1e-9; tc=X[t].mean(1)
    ar=np.linalg.norm(np.cross(X[t[:,1]]-X[t[:,0]],X[t[:,2]]-X[t[:,0]]),axis=1)/2
    ok=(tn[:,1]>0.25)&(tc[:,1]>np.percentile(X[:,1],35)); pr=ar*ok; pr/=pr.sum()
    from recolor11 import floret,sprig
    r=0.0085; PP=[];CC=[];TT=[]; base=0
    for k,ti in enumerate(rng.choice(len(t),NF+NL,p=pr)):
        u,v=rng.random(2)
        if u+v>1: u,v=1-u,1-v
        q=X[t[ti,0]]+u*(X[t[ti,1]]-X[t[ti,0]])+v*(X[t[ti,2]]-X[t[ti,0]]); nn=tn[ti]; q=q+nn*r*rng.uniform(0.8,1.8)
        lit=0.86+0.14*max(0,nn@LDIR)
        if k<NF: pp,cc,tt=floret(q,nn,r*rng.uniform(0.75,1.3),rng,np.array([255,252,244.])*lit,np.array([240,212,110.])*lit)
        else: pp,cc,tt=sprig(q,nn,r*0.85,rng,np.array([200,168,226.])*lit)
        PP.append(pp);CC.append(cc);TT.append(tt+base); base+=len(pp)
    add_prim(g,np.vstack(PP),s2l(np.clip(np.vstack(CC),0,252)),np.vstack(TT),'FillerFlowers')
    g.save(dst,{})
    print('잔꽃',NF,'라벤더',NL,'추가 정점',base)
if __name__=='__main__':
    a=sys.argv[1:]; run('web/FlowerBank_1m_sample_A_1002_v15.glb',AMB=float(a[0]),LEAFMUL=tuple(map(float,a[1].split(','))))
