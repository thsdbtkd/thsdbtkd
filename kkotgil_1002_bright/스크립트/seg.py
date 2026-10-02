import numpy as np,colorsys
from glbio import load
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def seg(f):
    d=load(f); t=d['idx'].reshape(-1,3); n=len(d['pos'])
    r=np.concatenate([t[:,0],t[:,1],t[:,2]]); c=np.concatenate([t[:,1],t[:,2],t[:,0]])
    k,lab=connected_components(coo_matrix((np.ones(len(r)),(r,c)),shape=(n,n)),directed=False)
    col=np.clip(d['col'],0,1)
    mx=col.max(1); mn=col.min(1); v=mx; s=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0)
    hsv=np.array([colorsys.rgb_to_hsv(*x) for x in col]); h=hsv[:,0]*360
    sizes=np.bincount(lab)
    # component mean color -> hue
    cm=np.stack([np.bincount(lab,weights=col[:,i])/sizes for i in range(3)],1)
    ch=np.array([colorsys.rgb_to_hsv(*x)[0]*360 for x in cm])
    cmn=np.full((k,3),1e9); cmx=np.full((k,3),-1e9); np.minimum.at(cmn,lab,d['pos']); np.maximum.at(cmx,lab,d['pos'])
    ext=(cmx-cmn).max(1); cen=(cmn+cmx)/2
    return dict(d=d,lab=lab,k=k,sizes=sizes,ch=ch,ext=ext,cen=cen,cm=cm,h=h)
