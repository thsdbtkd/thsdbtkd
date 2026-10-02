import sys,numpy as np
import recolor6_base as R
from glbio import load,save
from recolor5 import s2l,l2s
WARM=np.array([11,4,-11.])
GAP=np.array([110,150,80.])
LIFT=5
# 해: 촬영 화면 기준 왼쪽 위 45°, 카메라 쪽으로 조금 (bank 는 −z 쪽에서, mound_c 는 +z 쪽에서 찍음)
SUN={'vr27n_D_bank':np.array([1,1.41,-0.5]),'vr27n_D_mound_c':np.array([-1,1.41,0.5])}
def vnormals(P,t):
    fn=np.cross(P[t[:,1]]-P[t[:,0]],P[t[:,2]]-P[t[:,0]]); n=np.zeros_like(P)
    for k in range(3): np.add.at(n,t[:,k],fn)
    return n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-12)
def run(src,seed,HL=12,DK=8,GAPW=1.0):
    d=load(src+'.glb'); P=d['pos']; t=d['idx'].reshape(-1,3)
    _,flower,vb=R.run(src+'.glb',None,seed,TOP=242,ret=True)
    v6=load('web/'+src+'_bright_v6.glb'); sg=l2s(v6['col'])
    n=vnormals(P,t); mn,mx=P.min(0),P.max(0); core=np.array([(mn[0]+mx[0])/2,mn[1],(mn[2]+mx[2])/2])
    ref=core.copy()[None].repeat(len(P),0)
    for b in np.unique(vb[flower]):
        m=vb==b; ref[m]=P[m].mean(0)-0.3*(P[m].mean(0)-core)   # 송이 중심 약간 안쪽
    flip=((P-ref)*n).sum(1)<0; n[flip]*=-1                          # 바깥을 향하게
    L=SUN[src]/np.linalg.norm(SUN[src]); nd=n@L
    # 1) 따뜻한 햇살
    sg=sg+WARM[None]; sg[flower]+=LIFT
    # 2) 꽃잎 빛: 해를 향한 면(nd>0.5) 하이라이트 +HL(따뜻하게), 반대쪽(nd<0) 최대 −DK
    hl=np.clip((nd-0.5)/0.5,0,1)[:,None]*np.array([HL,HL*0.85,HL*0.55])[None]
    dk=np.clip(-nd,0,1)[:,None]*DK
    sg[flower]=sg[flower]+hl[flower]-dk[flower]
    sg[~flower]=sg[~flower]+0.6*hl[~flower]-dk[~flower]
    # 3) 틈 밝히기: 밝기 하위 10% → 따뜻한 녹색 쪽으로
    br=sg.mean(1); p10=np.percentile(br,10); lowt=GAP.mean()
    w=np.clip((lowt-br)/max(lowt-br.min(),1),0,1)*GAPW; w[br>max(p10,lowt)]=0
    sg=sg*(1-w[:,None])+GAP[None]*w[:,None]
    sg[flower]=np.minimum(sg[flower],246); sg[~flower]=np.minimum(sg[~flower],232)
    out=s2l(np.clip(sg,0,255)); save(v6,out,'web/'+src+'_bright_v7.glb')
    print(src,'해 향한 면 비율 %.2f, 반대 %.2f, 틈 끌어올린 정점 %d'%((nd>0.5).mean(),(nd<0).mean(),(w>0).sum()))
if __name__=='__main__':
    run('vr27n_D_bank',7); run('vr27n_D_mound_c',11)
