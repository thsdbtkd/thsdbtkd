import io,json,numpy as np
from scipy.cluster.vq import kmeans2
from fbio import G
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
HYD5=[(232,240,224),(238,239,232),(228,238,216)]   # 흰·연초록만(라벤더·하늘색 제거)
HW=np.array([.5,.3,.2])
EMI=0.13
RS=1.0
def warm(v,dr=4,db=4):
    """대표값 v(sRGB)에서 R+dr, B-db 가 되도록 선형 배수"""
    return s2l(v[0]+dr)/s2l(v[0]), s2l(v[2]-db)/s2l(v[2])
def run(src_v4,orig,dst,K=1.15,seed=5):
    rng=np.random.default_rng(seed); g=G(src_v4); go=G(orig); j=g.j
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    Po={go.j['materials'][p['material']]['name']:p for p in go.j['meshes'][0]['primitives']}
    # 수국: 원본 정점 밝기를 그늘로, 색은 흰·연초록
    p=P['Hydrangea_1']; X=g.get(p['attributes']['POSITION']); C=g.get(p['attributes']['COLOR_0'])
    Co=go.get(Po['Hydrangea_1']['attributes']['COLOR_0'])
    _,l=kmeans2(X,8,minit='++',seed=seed); lum=Co[:,:3]@[0.2126,0.7152,0.0722]; s=np.clip(lum/np.percentile(lum,80),0,1)
    pal=np.array([s2l(HYD5[rng.choice(3,p=HW)]) for _ in range(8)])
    C[:,:3]=np.clip(pal[l]*K*(0.45+0.55*s)[:,None],0,1); g.put(p['attributes']['COLOR_0'],C)
    # 따뜻한 햇살: 꽃(대표 225,222,215)·잎(대표 91,144,69)
    fr,fb=warm((225,222,215)); lr,lb=warm((91,144,69))
    for nm,(r,b) in [('Flowers',(fr,fb)),('PetalEdge',(fr,fb)),('Hydrangea_1',(fr,fb)),('HydrangeaCore',(fr,fb)),('Leaf',(lr,lb)),('LeafCard',(lr,lb)),('Base',(lr,lb))]:
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); C[:,:3]*=(RS if nm in ('Flowers','PetalEdge','Hydrangea_1') else 1.0); C[:,0]*=r; C[:,2]*=b; C[:,:3]=np.clip(C[:,:3],0,1); g.put(p['attributes']['COLOR_0'],C)
    # 흰 장미가 회색으로 보이지 않게: 꽃 발광(무채색 텍스처) 0.13 → EMI
    for m in j['materials']:
        if m['name'] in ('Flowers','PetalEdge'): m['emissiveFactor']=[EMI*1.02,EMI,EMI*0.97]; m['pbrMetallicRoughness']['roughnessFactor']=0.95  # 반사 하이라이트로 하얗게 날아가는 것 방지
    g.save(dst,{}); print('warm 꽃 R×%.3f B×%.3f / 잎 R×%.3f B×%.3f'%(fr,fb,lr,lb))
if __name__=='__main__':
    import sys; EMI=float(sys.argv[1]) if len(sys.argv)>1 else 0.13; RS=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
    run('web/FlowerBank_1m_sample_A_1002_v4.glb','FlowerBank_1m_sample_A.glb','web/FlowerBank_1m_sample_A_1002_v5.glb')
