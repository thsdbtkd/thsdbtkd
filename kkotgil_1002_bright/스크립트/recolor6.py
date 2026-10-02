import sys,numpy as np
import recolor6_base as R
from recolor5 import write_backface,s2l,l2s
from glbio import load
from seg import seg
from holes2 import band_holes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
LEAF_A=np.array([91,135,67.]); LEAF_B=np.array([120,165,85.])
def depth_shade(P,flower,vb,col,DROP):
    """송이마다 바깥(중심·윗면) → 꽃 속 안쪽으로 어둡게: 바깥 1.0, 가장 안쪽 (1-DROP)"""
    mn,mx=P.min(0),P.max(0); core=np.array([(mn[0]+mx[0])/2,mn[1],(mn[2]+mx[2])/2])
    f=np.ones(len(P))
    for b in np.unique(vb[flower]):
        m=vb==b; c=P[m].mean(0); n=c-core; n[0]*=0.5; n/=np.linalg.norm(n)+1e-9
        h=(P[m]-c)@n; r=np.linalg.norm((P[m]-c)-np.outer(h,n),axis=1)
        hn=(h-h.min())/max(np.ptp(h),1e-9)                 # 1 = 바깥(윗면)
        rn=r/max(np.percentile(r,95),1e-9)                  # 1 = 가장자리
        out=np.clip(0.6*hn+0.4*np.clip(rn,0,1),0,1)         # 꽃 속 = 축 가까이·깊은 곳
        out=np.clip((out-0.25)/0.5,0,1); f[m]=(1-DROP)+DROP*out**0.7
    return f
def band_noise(d,leaf,col,seed):
    rng=np.random.default_rng(seed); P=d['pos']; t=d['idx'].reshape(-1,3)
    key=np.round(P,5); _,uid=np.unique(key,axis=0,return_inverse=True); uid=uid.ravel(); nu=uid.max()+1; tu=uid[t]
    r=np.r_[tu[:,0],tu[:,1],tu[:,2]]; c=np.r_[tu[:,1],tu[:,2],tu[:,0]]
    k,cl=connected_components(coo_matrix((np.ones(len(r)),(r,c)),shape=(nu,nu)),directed=False)
    sz=np.bincount(cl); vcomp=cl[uid]; big=(sz[vcomp]>=40)&leaf    # 큰 초록 판(띠)
    # 같은 위치 정점은 같은 값(이음매 없이), 잎 색 두 끝을 섞고 밝기 ±12
    mix=rng.uniform(0,1,nu)[uid]; nz=rng.uniform(-12,12,nu)[uid]
    sg=LEAF_A[None]+(LEAF_B-LEAF_A)[None]*mix[:,None]+nz[:,None]
    col[big]=s2l(np.clip(sg[big],0,220)); return big.sum()
def run(src,seed,AO,CREV,PN,WG,DROP,WARM=4):
    R.WGAIN=WG
    col,flower,vb=R.run(src+'.glb',None,seed,AO=AO,CREV=CREV,TOP=242,PN=PN,ret=True)
    d=load(src+'.glb'); v4=load('web/'+src+'_bright_v4.glb'); leaf=~flower
    col[leaf]=v4['col'][leaf]
    col[flower]*=depth_shade(d['pos'],flower,vb,col,DROP)[flower,None]
    S=seg(src+'.glb'); lf=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    nb=band_noise(d,lf,col,seed) if 'bank' in src else 0
    sg=l2s(col); sg[:,0]+=WARM; sg[:,2]-=WARM
    sg[flower]=np.minimum(sg[flower],242); sg[leaf]=np.minimum(sg[leaf],220)
    col=s2l(np.clip(sg,0,255)); extra,info=band_holes(d,lf)
    write_backface(d,col,'web/'+src+'_bright_v6.glb',extra); print(src,'띠 노이즈 정점',nb,'구멍',len(info))
if __name__=='__main__':
    a=list(map(float,sys.argv[1:6]))
    run('vr27n_D_bank',7,*a); run('vr27n_D_mound_c',11,*a)
