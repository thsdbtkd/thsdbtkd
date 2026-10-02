"""샘플 꽃둑 17판: 원래 장미 형상을 빼고 절차적 작약 송이로 교체(목표 사진의 작약처럼).
잎·잔꽃·라벤더는 15판 그대로. 작약은 조명 없는 재질, 빛·그늘은 정점색에 구움."""
import sys,numpy as np
from scipy.cluster.vq import kmeans2
from fbio import G
from sampleA_v15 import add_prim
from peony import peony
PAL=[((255,236,222),(244,176,156),.30),((253,216,204),(236,144,144),.22),((248,182,186),(226,108,128),.16),
     ((240,132,148),(206,70,100),.10),((255,226,200),(240,164,136),.22)]
def run(src,dst,SIZE=1.15,seed=17):
    rng=np.random.default_rng(seed); g=G(src); j=g.j; prims=j['meshes'][0]['primitives']
    name=lambda p: j['materials'][p['material']]['name']; P={name(p):p for p in prims}
    X=[];N=[]
    for n in ('Flowers','PetalEdge'):
        X.append(g.get(P[n]['attributes']['POSITION'])); N.append(g.get(P[n]['attributes']['NORMAL']))
    X=np.vstack(X); N=np.vstack(N); cen,lab=kmeans2(X,36,minit='++',seed=5,iter=40)
    hp=g.get(P['Hydrangea_1']['attributes']['POSITION']); hc=hp.mean(0); hr=np.ptp(hp,0)
    core=np.array([X[:,0].mean(),X[:,1].min()-0.05,X[:,2].mean()])
    blooms=[]
    for b in range(36):
        m=lab==b
        if m.sum()<80: continue
        ext=np.ptp(X[m],0); R=0.5*np.sort(ext)[-2]*SIZE
        ax=cen[b]-core; ax[1]=max(ax[1],0.05); ax=ax/np.linalg.norm(ax); ax=0.5*ax+0.5*np.array([0,1,0.]); ax/=np.linalg.norm(ax)
        blooms.append((cen[b],ax,R))
    Rm=np.median([b[2] for b in blooms])
    for dx,dz in [(-0.18,-0.12),(0.2,0.05),(-0.1,0.28)]:
        blooms.append((hc+np.array([dx*hr[0],0.0,dz*hr[2]]),np.array([0,1,0.]),Rm*rng.uniform(0.9,1.1)))
    LX=g.get(P['Leaf']['attributes']['POSITION'])
    w=np.array([p[2] for p in PAL]); PP=[];CC=[];TT=[];nv=0
    for c,ax,R in blooms:
        k=rng.choice(len(PAL),p=w/w.sum()); base,deep,_=PAL[k]
        # 떠 있지 않게: 송이 바로 아래 잎 표면 높이에 꽃 밑동을 붙임
        d2=((LX[:,[0,2]]-np.array(c)[[0,2]])**2).sum(1); near=LX[d2<(R*1.2)**2]
        c=np.array(c,float)
        if len(near): c[1]=min(c[1],np.percentile(near[:,1],80)+R*0.15)
        p,cl,t=peony(c,ax,R,base,rng,deep_rgb=deep)
        PP.append(p);CC.append(cl);TT.append(t+nv); nv+=len(p)
    j['meshes'][0]['primitives']=[p for p in prims if name(p) not in ('Flowers','PetalEdge','Hydrangea_1','HydrangeaCore')]
    add_prim(g,np.vstack(PP),np.vstack(CC),np.vstack(TT),'Peony')
    g.save(dst,{})
    print('작약',len(blooms),'정점',nv,'삼각형',len(np.vstack(TT)))
if __name__=='__main__':
    run('web/FlowerBank_1m_sample_A_1002_v15.glb','web/FlowerBank_1m_sample_A_1002_v17.glb',float(sys.argv[1]))
