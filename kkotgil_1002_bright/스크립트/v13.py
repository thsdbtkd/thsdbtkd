import sys,numpy as np
from glbio import load,save
from seg import seg
from recolor5 import s2l,l2s,write_backface
from holes2 import band_holes
CREAM=np.array([236,232,218.]); IVORY=np.array([244,232,208.]); LEAF=np.array([127,174,85.])
def planA(src,HG,LG):
    d=load(src+'.glb'); S=seg(src+'.glb'); flower=np.load('flower_'+src+'.npy'); lv=~flower
    cat=np.load(src+'_cat9.npy'); hyd=((S['sizes']==6)[S['lab']])&flower
    v=l2s(load('web/'+src+'_bright_v12A.glb')['col'])
    out=v.copy()
    # 1) 수국 자리: 크림 흰색, 명암 L^γ 를 L^HG 로 완화(그대로 둔 9판 명암에서 L 역산)
    Lg=np.clip(v[hyd].max(1)/238.0,0,1); L=Lg**(1/1.1)
    out[hyd]=CREAM[None]*(L**HG)[:,None]
    # 2) 모브 장미(연핑크 송이): 아이보리로, 명암은 그대로
    mauve=(cat==3)&~hyd&(v[:,2]>=v[:,1]-8)
    out[mauve]=IVORY[None]*np.clip(v[mauve,0]/242.0,0,1)[:,None]
    # 3) 잎·틈: (127,174,85) 쪽, 원본 밝기 비율 유지
    Lin=np.clip(d['col'],0,1)@[0.2126,0.7152,0.0722]; Ll=Lin[lv]; Lm=np.median(Ll)
    out[lv]=LEAF[None]*np.clip((Ll/Lm)**0.9*LG,0,1.6)[:,None]
    out=np.clip(out,0,240)
    lf=((S['ch']>=60)&(S['ch']<170))[S['lab']]; extra,_=band_holes(d,lf)
    write_backface(d,s2l(out),'web/'+src+'_bright_v13A.glb',extra)
    m=np.zeros((len(out),3)); m[hyd]=1; write_backface(d,m,'web/maskH13_'+src+'.glb',extra)
    print(src,'수국 정점',hyd.sum(),'모브→아이보리 정점',mauve.sum())
if __name__=='__main__':
    HG,LG=float(sys.argv[1]),float(sys.argv[2])
    planA('vr27n_D_bank',HG,LG); planA('vr27n_D_mound_c',HG,LG)
