"""11판: 참고 사진보다 낫게 — 10판 색 + (1) 흰 잔꽃(안개꽃 같은 작은 꽃) 심기 (2) 라벤더 잔꽃 (3) 잎 올리브·노란 기
(4) 위·바깥쪽 따뜻한 햇살 빛(역광 느낌) (5) 공중에 날리는 꽃잎은 따로 파일(합성용 겹)."""
import sys,json,struct,numpy as np
import recolor10 as R10
from glbio import load
from seg import seg
from recolor5 import s2l,l2s
from holes2 import band_holes
R10.HYD=[((255,250,242),.65),((248,210,222),.30),((226,198,234),.05)]   # 수국 자리는 흰 잔꽃 위주
R10.ROSE=[((252,220,212),.25),((254,240,230),.08),((248,190,180),.20),((242,146,170),.32),((230,100,134),.15)]
R10.LEAF=np.array([146,160,70.])                                      # 참고 사진 잎: 올리브 연두
GLOW=np.array([255,238,220.])
def vnormals(P,t):
    fn=np.cross(P[t[:,1]]-P[t[:,0]],P[t[:,2]]-P[t[:,0]]); n=np.zeros_like(P)
    for k in range(3): np.add.at(n,t[:,k],fn)
    return n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-12)
def floret(c,n,r,rng,petal_col,center_col,npet=5):
    """작은 꽃 하나: 꽃잎 npet 장(삼각형 2개씩) + 가운데"""
    a=np.cross(n,[0,1,0]); a=a/np.linalg.norm(a) if np.linalg.norm(a)>1e-6 else np.array([1,0,0.]); b=np.cross(n,a)
    P=[c+n*r*0.15]; C=[center_col]; T=[]
    th0=rng.uniform(0,2*np.pi)
    for k in range(npet):
        th=th0+2*np.pi*k/npet; d=np.cos(th)*a+np.sin(th)*b; s=np.cos(th+np.pi/2)*a+np.sin(th+np.pi/2)*b
        i=len(P)
        P+= [c+d*r*0.25+s*r*0.28, c+d*r*1.0+s*r*0.32+n*r*0.1, c+d*r*1.05-s*r*0.32+n*r*0.1, c+d*r*0.25-s*r*0.28]
        sh=0.9+0.1*np.cos(th-th0)
        C+= [petal_col*sh]*4
        T+= [[i,i+1,i+2],[i,i+2,i+3],[0,i,i+3]]
    return np.array(P),np.array(C),np.array(T)
def sprig(c,n,r,rng,col):
    """라벤더 잔꽃: 위로 뻗은 짧은 줄기에 작은 사각 꽃 6개"""
    up=n*0.5+np.array([0,1,0.])*0.5; up/=np.linalg.norm(up); P=[];C=[];T=[]
    for k in range(6):
        p=c+up*r*k*0.9+rng.normal(0,r*0.15,3); s=r*0.45*(1-k/9)
        a=np.cross(up,[1,0,0]); a/=np.linalg.norm(a)+1e-9; b=np.cross(up,a); i=len(P)
        P+=[p+a*s,p+b*s,p-a*s,p-b*s]; C+=[col*(0.85+0.15*k/6)]*4; T+=[[i,i+1,i+2],[i,i+2,i+3]]
    return np.array(P),np.array(C),np.array(T)
def write(d,P,col,t,out,double=True):
    j=json.loads(json.dumps(d['json']))
    inter=np.hstack([P,col]).astype(np.float32); idx=np.vstack([t,t[:,[0,2,1]]]) if double else t
    idx=idx.astype(np.uint32).ravel(); vb=inter.tobytes(); ib=idx.tobytes()
    pad=lambda b:b+b'\0'*((4-len(b)%4)%4); binb=pad(vb)+pad(ib)
    j['bufferViews'][0].update(byteOffset=0,byteLength=len(vb)); j['bufferViews'][1].update(byteOffset=len(pad(vb)),byteLength=len(ib))
    j['accessors'][0].update(count=len(P),min=P.min(0).tolist(),max=P.max(0).tolist()); j['accessors'][1]['count']=len(P)
    j['accessors'][2].update(count=len(idx),componentType=5125); j['buffers'][0]['byteLength']=len(binb)
    for m in j['materials']: m['doubleSided']=False
    js=json.dumps(j,separators=(',',':')).encode(); js+=b' '*((4-len(js)%4)%4)
    open(out,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb)
SUN={'vr27n_D_bank':np.array([0.6,1,-0.8]),'vr27n_D_mound_c':np.array([-0.6,1,0.8])}   # 촬영 앞면 위쪽에서 비추는 햇살
def run(src,seed,NF,NL,out):
    rng=np.random.default_rng(seed+100)
    col,flower=R10.run(src,seed)
    d=load(src+'.glb'); P=d['pos'].astype(float); t=d['idx'].reshape(-1,3).astype(np.int64)
    n=vnormals(P,t); mn,mx=P.min(0),P.max(0); core=np.array([(mn[0]+mx[0])/2,mn[1]-0.1*(mx[1]-mn[1]),(mn[2]+mx[2])/2])
    out_dir=P-core; out_dir/=np.linalg.norm(out_dir,axis=1,keepdims=True)
    flip=(n*out_dir).sum(1)<0; n[flip]*=-1
    # (4) 햇살 빛: 해 쪽을 향한 면을 따뜻한 빛으로 (꽃은 최대 30%, 잎은 노란 기)
    L=SUN[src]/np.linalg.norm(SUN[src]); nd=np.clip(n@L,0,1)
    sg=l2s(col); w=(nd**2)[:,None]
    sg[flower]=sg[flower]*(1-0.30*w[flower])+GLOW*0.30*w[flower]
    sg[~flower]=sg[~flower]*(1-0.35*w[~flower])+np.array([196,196,96.])*0.35*w[~flower]
    col=s2l(np.clip(sg,0,252))
    # 띠 빈칸 채움
    S=seg(src+'.glb'); lf=((S['ch']>=60)&(S['ch']<170))[S['lab']]; extra,_=band_holes(d,lf)
    t=np.vstack([t,extra.astype(np.int64)]) if len(extra) else t
    # (1)(2) 잔꽃 심기: 바깥을 향한 잎 삼각형 위에 면적 비례로
    lt=t[(~flower[t]).all(1)]
    ar=np.linalg.norm(np.cross(P[lt[:,1]]-P[lt[:,0]],P[lt[:,2]]-P[lt[:,0]]),axis=1)/2
    tn=n[lt].mean(1); tn/=np.linalg.norm(tn,axis=1,keepdims=True)+1e-9; tc=P[lt].mean(1)
    ok=((tn*((tc-core)/np.linalg.norm(tc-core,axis=1,keepdims=True))).sum(1)>0.2)&(ar<np.percentile(ar,99))
    pr=ar*ok; pr/=pr.sum()
    size=np.linalg.norm(mx-mn); r=size*0.0085
    addP=[];addC=[];addT=[]; base=len(P)
    for k,ti in enumerate(rng.choice(len(lt),NF+NL,p=pr)):
        u,v=rng.random(2)
        if u+v>1: u,v=1-u,1-v
        q=P[lt[ti,0]]+u*(P[lt[ti,1]]-P[lt[ti,0]])+v*(P[lt[ti,2]]-P[lt[ti,0]])
        nn=tn[ti]; q=q+nn*r*rng.uniform(0.8,2.0)
        lit=0.80+0.20*max(0,nn@L)
        if k<NF:
            pc=np.array([252,250,244.])*lit*rng.uniform(0.95,1.0); cc=np.array([236,214,120.])*lit
            pp,cc_,tt=floret(q,nn,r*rng.uniform(0.8,1.25),rng,pc,cc)
        else:
            pp,cc_,tt=sprig(q,nn,r*0.9,rng,np.array([196,168,222.])*lit)
        addT.append(tt+base); addP.append(pp); addC.append(cc_); base+=len(pp)
    P2=np.vstack([P]+addP); C2=np.vstack([col,s2l(np.clip(np.vstack(addC),0,252))]); T2=np.vstack([t]+addT)
    write(d,P2,C2,T2,out)
    print(src,'잔꽃',NF,'라벤더',NL,'정점',len(P2))
    return P,mn,mx
def petals(src,seed,N,out,box=None,front=None):
    """공중에 날리는 꽃잎(합성용 겹): 앞면 앞쪽 공간에 N장. box=(mn,mx) 를 주면 그 상자 기준(샘플 A)"""
    rng=np.random.default_rng(seed+7); d=load(src+'.glb'); P=d['pos']; mn,mx=(P.min(0),P.max(0)) if box is None else box; sz=mx-mn
    if front is None: front=-1 if 'bank' in src else 1
    pal=[(250,214,206),(246,190,186),(240,160,176),(254,236,226)]
    PP=[];CC=[];TT=[]
    for k in range(N):
        c=np.array([rng.uniform(mn[0]-0.1*sz[0],mx[0]+0.1*sz[0]),rng.uniform(mn[1]+0.2*sz[1],mx[1]+0.6*sz[1]),
                    (mx[2] if front>0 else mn[2])+front*rng.uniform(0.05,0.9)*max(sz[0],sz[2])*0.5])
        r=np.linalg.norm(sz)*rng.uniform(0.006,0.011); a=rng.normal(size=3); a/=np.linalg.norm(a); b=np.cross(a,rng.normal(size=3)); b/=np.linalg.norm(b)
        i=len(PP); col=np.array(pal[rng.integers(4)],float)*rng.uniform(0.9,1.0)
        PP+=[c-a*r, c-b*r*0.6+a*r*0.2, c+a*r, c+b*r*0.6+a*r*0.2]; CC+=[col*0.92,col,col,col*0.97]; TT+=[[i,i+1,i+2],[i,i+2,i+3]]
    PP=np.array(PP); CC=s2l(np.array(CC)); TT=np.array(TT)
    if box is not None: d=dict(d); d['json']=json.loads(json.dumps(d['json'])); [nd.pop('scale',None) for nd in d['json']['nodes']]
    write(d,PP,CC,TT,out)
if __name__=='__main__':
    run('vr27n_D_bank',7,1300,200,'web/vr27n_D_bank_bright_v11.glb'); petals('vr27n_D_bank',7,160,'web/petals_bank_v11.glb')
    run('vr27n_D_mound_c',11,850,140,'web/vr27n_D_mound_c_bright_v11.glb'); petals('vr27n_D_mound_c',11,110,'web/petals_mound_c_v11.glb')
    from fbio import G
    g=G('FlowerBank_1m_sample_A.glb'); X=np.vstack([g.get(p['attributes']['POSITION']) for p in g.j['meshes'][0]['primitives']])
    petals('vr27n_D_bank',5,90,'web/petals_A_v11.glb',box=(X.min(0),X.max(0)),front=1)
