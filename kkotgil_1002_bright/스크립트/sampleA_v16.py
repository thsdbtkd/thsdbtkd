"""샘플 꽃둑 16판: 15판 + 형태를 목표 사진 쪽으로
- 장미(작약처럼) 송이를 1.3배 키움(송이 중심 기준)
- 가운데 큰 수국 덩어리를 빼고, 그 자리와 잎만 보이는 빈자리에 장미 송이를 복제해 채움(목표 사진처럼 빽빽하게)"""
import sys,numpy as np
from scipy.cluster.vq import kmeans2
from fbio import G
from sampleA_v15 import add_prim
def addv(g,arr,target=34962):
    j=g.j
    while len(g.bin)%4: g.bin.append(0)
    off=len(g.bin); g.bin+=arr.tobytes(); j['bufferViews'].append({'buffer':0,'byteOffset':off,'byteLength':arr.nbytes,'target':target}); return len(j['bufferViews'])-1
def acc(g,arr,typ):
    j=g.j; arr=np.ascontiguousarray(arr.astype(np.float32)); a={'bufferView':addv(g,arr),'componentType':5126,'count':len(arr),'type':typ}
    if typ=='VEC3': a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
    j['accessors'].append(a); return len(j['accessors'])-1
def run(src,dst,SCALE=1.3,NEXTRA=14,seed=8):
    rng=np.random.default_rng(seed); g=G(src); j=g.j; prims=j['meshes'][0]['primitives']
    name=lambda p: j['materials'][p['material']]['name']
    P={name(p):p for p in prims}
    D={}
    for n in ('Flowers','PetalEdge'):
        p=P[n]; D[n]=dict(pos=g.get(p['attributes']['POSITION']),nor=g.get(p['attributes']['NORMAL']),uv=g.get(p['attributes']['TEXCOORD_0']),col=g.get(p['attributes']['COLOR_0']),idx=g.get(p['indices']).astype(np.int64).reshape(-1,3),mat=p['material'])
    allp=np.vstack([D[n]['pos'] for n in D]); cen,_=kmeans2(allp[::5],36,minit='++',seed=5,iter=40)
    for n in D:
        D[n]['lab']=np.argmin(((D[n]['pos'][:,None]-cen[None])**2).sum(-1),1)
    # 1) 송이 키우기
    for n in D:
        c=cen[D[n]['lab']]; D[n]['pos']=c+(D[n]['pos']-c)*SCALE
        g.put(P[n]['attributes']['POSITION'],D[n]['pos'])
        a=j['accessors'][P[n]['attributes']['POSITION']]; a['min']=D[n]['pos'].min(0).tolist(); a['max']=D[n]['pos'].max(0).tolist()
    # 1b) 작약처럼: 송이 가운데는 크림, 바깥 꽃잎은 원래 분홍 — 밝기는 그대로, 위를 향한 면은 따뜻한 빛 +
    def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
    cream=s2l((255,238,222)); lw=np.array([0.2126,0.7152,0.0722])
    for n in D:
        p=D[n]['pos']; c=cen[D[n]['lab']]; r=np.linalg.norm((p-c)[:,[0,2]],axis=1)
        rr=np.zeros(len(p))
        for b in np.unique(D[n]['lab']):
            m=D[n]['lab']==b; rr[m]=r[m]/max(np.percentile(r[m],90),1e-6)
        w=np.clip(rr,0,1)**0.9
        col=D[n]['col'].copy(); L=col[:,:3]@lw
        cc=cream[None]*(L/(cream@lw))[:,None]
        col[:,:3]=col[:,:3]*w[:,None]+cc*(1-w[:,None])*0.22+col[:,:3]*(1-w[:,None])*0.78
        up=np.clip(D[n]['nor'][:,1],0,1)
        col[:,:3]=np.clip(col[:,:3]*1.07*(1+0.22*up[:,None]*np.array([1.0,0.97,0.9])),0,1)
        D[n]['col']=col; g.put(P[n]['attributes']['COLOR_0'],col)
    # 2) 수국 빼기 (수국·수국 속 primitive 제거)
    hp=g.get(P['Hydrangea_1']['attributes']['POSITION']); hc=hp.mean(0); hr=np.ptp(hp,0)
    j['meshes'][0]['primitives']=[p for p in prims if name(p) not in ('Hydrangea_1','HydrangeaCore')]
    # 3) 빈자리 찾기: 잎 윗면 중 송이 중심에서 먼 곳 + 수국 자리
    lp=P['Leaf']; LX=g.get(lp['attributes']['POSITION']); LN=g.get(lp['attributes']['NORMAL'])
    top=LX[(LN[:,1]>0.3)&(LX[:,1]>np.percentile(LX[:,1],45))]
    rad=np.median([np.ptp(D['Flowers']['pos'][D['Flowers']['lab']==b],0).max() for b in range(36) if (D['Flowers']['lab']==b).sum()>50])*0.5
    spots=[hc+np.array([dx,0.02,dz])*hr for dx,dz in [(-0.22,-0.15),(0.2,-0.12),(0,0.2),(-0.15,0.22),(0.22,0.2)]]
    cands=top[rng.permutation(len(top))[:4000]]
    occupied=list(cen)+spots
    for q in cands:
        if len(spots)>=5+NEXTRA: break
        if min(np.linalg.norm(np.array(occupied)[:,[0,2]]-q[[0,2]],axis=1))>rad*1.5: spots.append(q+np.array([0,rad*0.3,0])); occupied.append(q)
    # 4) 장미 송이 복제해서 빈자리에 놓기
    good=[b for b in range(36) if (D['Flowers']['lab']==b).sum()>150]
    newgeo={n:dict(pos=[],nor=[],uv=[],col=[],idx=[],base=0) for n in D}
    for s in spots:
        b=good[rng.integers(len(good))]; th=rng.uniform(0,2*np.pi); R=np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]]); sc=rng.uniform(0.9,1.1)
        for n in D:
            m=D[n]['lab']==b; vid=np.where(m)[0]
            if len(vid)==0: continue
            remap=-np.ones(len(m),int); remap[vid]=np.arange(len(vid))
            tri=D[n]['idx'][m[D[n]['idx']].all(1)]
            pp=(D[n]['pos'][vid]-cen[b])@R.T*sc+s
            G_=newgeo[n]; G_['pos'].append(pp); G_['nor'].append(D[n]['nor'][vid]@R.T); G_['uv'].append(D[n]['uv'][vid]); G_['col'].append(D[n]['col'][vid]); G_['idx'].append(remap[tri]+G_['base']); G_['base']+=len(vid)
    added=0
    for n in D:
        G_=newgeo[n]
        if not G_['pos']: continue
        ip=acc(g,np.vstack(G_['pos']),'VEC3'); inn=acc(g,np.vstack(G_['nor']),'VEC3'); iu=acc(g,np.vstack(G_['uv']),'VEC2'); ic=acc(g,np.vstack(G_['col']),'VEC4')
        idx=np.vstack(G_['idx']).astype(np.uint32).ravel(); bv=addv(g,idx,34963); j['accessors'].append({'bufferView':bv,'componentType':5125,'count':len(idx),'type':'SCALAR'})
        j['meshes'][0]['primitives'].append({'attributes':{'POSITION':ip,'NORMAL':inn,'TEXCOORD_0':iu,'COLOR_0':ic},'indices':len(j['accessors'])-1,'material':D[n]['mat']})
        added+=len(idx)//3
    j['buffers'][0]['byteLength']=len(g.bin)
    g.save(dst,{})
    print('송이 배율',SCALE,'수국 삭제, 복제 송이',len(spots),'추가 삼각형',added)
if __name__=='__main__':
    run('web/FlowerBank_1m_sample_A_1002_v15.glb','web/FlowerBank_1m_sample_A_1002_v16.glb',float(sys.argv[1]),int(sys.argv[2]))
