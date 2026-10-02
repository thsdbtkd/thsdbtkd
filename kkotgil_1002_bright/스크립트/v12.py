import sys,numpy as np
from glbio import load,save
from recolor5 import s2l,l2s
LEAF=np.array([146,160,70.])
def planA(src,LG):
    d=load(src+'.glb'); flower=np.load('flower_'+src+'.npy'); lv=~flower
    Lin=np.clip(d['col'],0,1)@[0.2126,0.7152,0.0722]; Ll=Lin[lv]; Lm=np.median(Ll)
    v9=load('web/'+src+'_bright_v9.glb'); col=v9['col'].copy()
    col[lv]=s2l(np.clip(LEAF[None]*np.clip((Ll/Lm)**0.9*LG,0,1.6)[:,None],0,235))   # 원본 밝기 비율 유지
    save(v9,col,'web/'+src+'_bright_v12A.glb')
def planB():
    import recolor10 as R10, recolor11 as R11
    alt=np.random.default_rng(99); orig=R10.pick; cnt=[0]; rep=[0]
    def pick(rng,pal):
        t=orig(rng,pal)
        if np.allclose(t,(230,100,134)) or np.allclose(t,(232,108,142)): t=np.array((242,156,178.) if alt.random()<0.5 else (248,196,184.)); rep[0]+=1
        cnt[0]+=1 if t is not None else 0
        return t
    R10.pick=pick
    R11.run('vr27n_D_bank',7,1300,200,'web/vr27n_D_bank_bright_v12B.glb')
    R11.run('vr27n_D_mound_c',11,850,140,'web/vr27n_D_mound_c_bright_v12B.glb'); print('진분홍 → 분홍·피치로 바꾼 송이',rep[0],'/ 전체 고른 송이',cnt[0])
if __name__=='__main__':
    LG=float(sys.argv[1]); planA('vr27n_D_bank',LG); planA('vr27n_D_mound_c',LG)
    if len(sys.argv)>2: planB()
