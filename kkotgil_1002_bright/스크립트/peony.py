"""절차적 작약 송이: 바깥 큰 꽃잎(평평) → 안쪽 오목·돔 꽃잎 → 가운데 구겨진 잔꽃잎.
정점색에 따뜻한 빛(왼쪽 위)·밑동 그늘·꽃잎 가장자리 밝음을 구움(조명 없는 재질용)."""
import numpy as np
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
def frame(axis):
    a=axis/np.linalg.norm(axis); t=np.cross(a,[0,0,1.]) if abs(a[2])<0.9 else np.cross(a,[1,0,0.]); t/=np.linalg.norm(t); b=np.cross(a,t); return t,b,a
NU,NV=5,6
SH0=0.76
def petal(length,width,tilt,cup,curl,rng):
    P=[];V=[];U=[]
    rf=rng.uniform(0.5,1.0); ph=rng.uniform(0,6.28)
    for i in range(NV):
        v=i/(NV-1); w=width*np.sin(np.pi*(0.10+0.90*v))**0.5*(1.0 if v<0.8 else 1-((v-0.8)/0.2)**2*0.45)
        for k in range(NU):
            u=(k/(NU-1)-0.5)
            ru=0.07*rf*np.sin(u*10+ph)*v**2*length
            ang=tilt+curl*v
            x=v*length*np.cos(ang)-cup*(u*2)**2*w*0.6*np.sin(ang); z=v*length*np.sin(ang)+cup*(u*2)**2*w*0.6*np.cos(ang)+ru
            P.append([x,u*w*2,z]); V.append(v); U.append(u)
    T=[]
    for i in range(NV-1):
        for k in range(NU-1):
            a=i*NU+k; T+=[[a,a+1,a+NU+1],[a,a+NU+1,a+NU]]
    P=np.array(P); T=np.array(T)
    n=np.zeros_like(P); fn=np.cross(P[T[:,1]]-P[T[:,0]],P[T[:,2]]-P[T[:,0]])
    for q in range(3): np.add.at(n,T[:,q],fn)
    n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    if n[:,2].mean()<0: n=-n
    return P,np.array(V),np.array(U),T,n
RINGS=[(9,1.00,0.15,0.30,0.30,0.00),(9,0.90,0.40,0.55,0.40,0.12),(9,0.78,0.70,0.70,0.45,0.28),(8,0.66,0.98,0.80,0.45,0.45),
       (8,0.54,1.18,0.85,0.40,0.62),(10,0.42,1.32,0.85,0.30,0.78),(12,0.30,1.45,0.90,0.20,0.92),(10,0.20,1.50,0.90,0.10,1.0)]
def peony(center,axis,R,base_rgb,rng,light=np.array([-0.55,0.75,0.40]),deep_rgb=None,warm=np.array([255,244,226.])):
    t,b,a=frame(axis); L=light/np.linalg.norm(light)
    base=np.array(base_rgb,float); deep=np.array(deep_rgb if deep_rgb is not None else base*np.array([0.96,0.78,0.84]),float)
    P=[];C=[];T=[]; nv=0
    for ri,(n,lf,tilt,cup,curl,dep) in enumerate(RINGS):
        for j in range(n):
            phi=2*np.pi*j/n+ri*0.41+rng.normal(0,0.14)
            length=R*lf*rng.uniform(0.88,1.08); width=length*rng.uniform(0.45,0.58)*(1.2 if ri<2 else 1.0)
            pp,vv,uu,tt,nn=petal(length,width,tilt+rng.normal(0,0.09),cup,curl,rng)
            dr=np.cos(phi)*t+np.sin(phi)*b; tg=-np.sin(phi)*t+np.cos(phi)*b
            M=np.stack([dr,tg,a],0)                                   # 지역→세계
            W=center[None]+(pp+np.array([R*0.04*(1-lf),0,R*0.10*dep]))@M
            Nw=nn@M; ndl=np.clip(Nw@L,0,1)
            depth=np.clip(dep*0.75+(1-vv)*0.45,0,1)                    # 안쪽·밑동일수록 진하고 어둡게
            col=base[None]*(1-0.70*depth[:,None])+deep[None]*(0.70*depth[:,None])
            col=col*rng.uniform(0.97,1.03)
            shade=(SH0+0.42*ndl)*(1-0.42*depth*(1-vv*0.4))
            rim=np.clip((np.abs(uu)*2-0.6)/0.4,0,1)*vv*0.10+np.clip(vv-0.75,0,1)*0.35   # 꽃잎 가장자리·끝 밝음
            cl=s2l(col)*shade[:,None]*(1+rim[:,None])
            cl=cl+ (ndl**3*0.12*vv)[:,None]*s2l(warm)[None]              # 햇살 반짝임
            P.append(W); C.append(np.clip(cl,0,1)); T.append(tt+nv); nv+=len(W)
    return np.vstack(P),np.vstack(C),np.vstack(T)
