import numpy as np
from glbio import load
from seg import seg
from recolor5 import write_backface
from holes2 import band_holes
import recolor6_base as R
for s,sd in [('vr27n_D_bank',7),('vr27n_D_mound_c',11)]:
    d=load(s+'.glb'); S=seg(s+'.glb'); _,flower,vb=R.run(s+'.glb',None,sd,TOP=242,ret=True)
    lf=((S['ch']>=60)&(S['ch']<170))[S['lab']]; extra,_=band_holes(d,lf)
    col=np.zeros((len(d['pos']),3)); col[flower]=1
    write_backface(d,col,'web/maskF_'+s+'.glb',extra)
    np.save('flower_'+s+'.npy',flower)
