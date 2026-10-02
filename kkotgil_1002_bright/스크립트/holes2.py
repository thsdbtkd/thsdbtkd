import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def band_holes(d,leaf,minverts=40):
    """큰 초록 띠(격자) 안의 빠진 칸(경계선으로 둘러싸인 4각) 찾기 → 채울 삼각형(원래 정점 번호) 반환"""
    t=d['idx'].reshape(-1,3); P=d['pos']
    key=np.round(P,5); _,uid=np.unique(key,axis=0,return_inverse=True); uid=uid.ravel(); nu=uid.max()+1; tu=uid[t]
    r=np.r_[tu[:,0],tu[:,1],tu[:,2]]; c=np.r_[tu[:,1],tu[:,2],tu[:,0]]
    k,cl=connected_components(coo_matrix((np.ones(len(r)),(r,c)),shape=(nu,nu)),directed=False)
    sz=np.bincount(cl); rep=np.zeros(nu,int); rep[uid]=np.arange(len(uid))
    E=defaultdict(int); dirE=set()
    for a,b,cc in tu:
        for e in ((a,b),(b,cc),(cc,a)): E[tuple(sorted(e))]+=1; dirE.add(e)
    out=[]; info=[]
    for comp in np.where(sz>=minverts)[0]:
        vs=set(np.where(cl==comp)[0].tolist())
        if not leaf[rep[list(vs)]].all(): continue
        bnd=[e for e,n in E.items() if n==1 and e[0] in vs]
        adj=defaultdict(set)
        for a,b in bnd: adj[a].add(b); adj[b].add(a)
        allE=set(E.keys()); found=set()
        for a,b in bnd:
            for c_ in adj[b]-{a}:
                for d_ in adj[a]-{b}:
                    if d_!=c_ and tuple(sorted((c_,d_))) in [tuple(sorted(x)) for x in [(c_,d_)]] and (tuple(sorted((c_,d_))) in E) and E[tuple(sorted((c_,d_)))]==1:
                        q=(a,b,c_,d_)
                        if tuple(sorted((a,c_))) in allE or tuple(sorted((b,d_))) in allE: continue  # 이미 면이 있는 칸
                        key_=tuple(sorted(q))
                        if key_ in found: continue
                        found.add(key_)
                        # 감김: 기존 면이 a->b 방향이면 새 면은 b->a 방향
                        if (a,b) in dirE: tri=[(b,a,d_),(b,d_,c_)]
                        else: tri=[(a,b,c_),(a,c_,d_)]
                        for x in tri: out.append([rep[v] for v in x])
                        info.append(float(np.ptp(P[rep[list(q)]],0).max()))
    return np.array(out,dtype=np.int64).reshape(-1,3),info
