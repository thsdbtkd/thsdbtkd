"""18판 2단계: 수국 빼고 원본 작약 송이 복제로 채움(sampleA_v16), 흰 잔꽃·라벤더(조명 없는 작은 꽃)"""
import sys,numpy as np
import sampleA_v16 as V16
from fbio import G
from sampleA_v15 import add_prim
from recolor11 import floret,sprig
from recolor5 import s2l
LDIR=np.array([3,5,4.])/np.linalg.norm([3,5,4])
V16.run('web/_s19a.glb','web/_s19b.glb',float(sys.argv[1]),0)
g=G('web/_s19b.glb'); j=g.j; rng=np.random.default_rng(11)
P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
lp=P['Leaf']; X=g.get(lp['attributes']['POSITION']); N=g.get(lp['attributes']['NORMAL']); t=g.get(lp['indices']).astype(np.int64).reshape(-1,3)
tn=N[t].mean(1); tn/=np.linalg.norm(tn,axis=1,keepdims=True)+1e-9; tc=X[t].mean(1)
ar=np.linalg.norm(np.cross(X[t[:,1]]-X[t[:,0]],X[t[:,2]]-X[t[:,0]]),axis=1)/2
ok=(tn[:,1]>0.25)&(tc[:,1]>np.percentile(X[:,1],35)); pr=ar*ok; pr/=pr.sum()
NF,NL=int(sys.argv[2]),int(sys.argv[3]); r=0.0085; PP=[];CC=[];TT=[];base=0
for k,ti in enumerate(rng.choice(len(t),NF+NL,p=pr)):
    u,v=rng.random(2)
    if u+v>1: u,v=1-u,1-v
    q=X[t[ti,0]]+u*(X[t[ti,1]]-X[t[ti,0]])+v*(X[t[ti,2]]-X[t[ti,0]]); nn=tn[ti]; q=q+nn*r*rng.uniform(0.8,1.8)
    lit=0.86+0.14*max(0,nn@LDIR)
    if k<NF: pp,cc,tt=floret(q,nn,r*rng.uniform(0.75,1.3),rng,np.array([255,250,240.])*lit,np.array([240,206,110.])*lit)
    else: pp,cc,tt=sprig(q,nn,r*0.85,rng,np.array([206,172,228.])*lit)
    PP.append(pp);CC.append(cc);TT.append(tt+base); base+=len(pp)
add_prim(g,np.vstack(PP),s2l(np.clip(np.vstack(CC),0,252)),np.vstack(TT),'FillerFlowers')
g.save('web/FlowerBank_1m_sample_A_1002_v19.glb',{}); print('잔꽃',NF,'라벤더',NL)
