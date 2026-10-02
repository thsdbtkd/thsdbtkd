"""샘플 A 10판: 꽃(장미·수국)을 조명 없는 재질로 바꾸고, 빛을 정점색에 구워 참고 사진처럼 맑은 분홍·크림이 나오게.
잎·바닥은 7판(조명 받는 재질) 그대로, 색만 따뜻한 녹색 쪽으로."""
import sys,io,numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from fbio import G
from recolor10 import ROSE,HYD,pick
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
LEAFMUL=np.array([1.0,0.92,0.80])
LDIR=np.array([3,5,4.])/np.linalg.norm([3,5,4])     # 렌더 방향광과 같은 방향
SAT=0.6
def bake(t,shade):
    tl=s2l(t)[None]; return l2s(np.power(tl,1+SAT*(1-shade[:,None]))*np.power(shade[:,None],1.5))
def run(src,orig,dst,AMB,seed=5):
    rng=np.random.default_rng(seed); g=G(src); go=G(orig); j=g.j
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    Po={go.j['materials'][p['material']]['name']:p for p in go.j['meshes'][0]['primitives']}
    # 텍스처 평균(무채색 아틀라스) 보정값
    tex=np.asarray(Image.open(io.BytesIO(g.imgbytes(j['textures'][4]['source']))).convert('RGB')).astype(float)
    tmean=s2l(tex.mean((0,1))).mean()
    rp=['Flowers','PetalEdge']; X=np.vstack([g.get(P[n]['attributes']['POSITION']) for n in rp])
    cen,_=kmeans2(X[::5],36,minit='++',seed=seed,iter=40); pal=np.array([pick(rng,ROSE) for _ in range(36)])
    for n in rp+['Hydrangea_1']:
        p=P[n]; Xp=g.get(p['attributes']['POSITION']); N=g.get(p['attributes']['NORMAL']); C=g.get(p['attributes']['COLOR_0'])
        Co=go.get(Po[n]['attributes']['COLOR_0']); lo=Co[:,:3]@[0.2126,0.7152,0.0722]
        if n=='Hydrangea_1':
            _,l=kmeans2(Xp,8,minit='++',seed=seed); pp=np.array([pick(rng,HYD) for _ in range(8)])
        else:
            l=np.argmin(((Xp[:,None]-cen[None])**2).sum(-1),1); pp=pal
        ndl=np.abs(N@LDIR)                                 # 양면 꽃잎: 절댓값
        depth=np.zeros(len(Xp))
        for b in np.unique(l):
            m=l==b; depth[m]=np.clip(lo[m]/np.percentile(lo[m],85),0,1)
        shade=np.clip(AMB+(1-AMB)*ndl,0,1)*(0.55+0.45*depth**0.8)    # 빛 + 원본 송이 안쪽 그늘
        c=bake(pp[l]+rng.normal(0,1,(len(Xp),3)) ,shade) if False else bake_rows(pp[l],shade)
        C[:,:3]=np.clip(s2l(np.clip(c,0,250))/(tmean if n!='Hydrangea_1' else 1.0),0,1); g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        if m['name'] in ('Flowers','PetalEdge','Hydrangea_1','HydrangeaCore'):
            m.setdefault('extensions',{})['KHR_materials_unlit']={}; m.pop('emissiveFactor',None); m.pop('emissiveTexture',None); m.pop('occlusionTexture',None)
    if 'KHR_materials_unlit' not in j['extensionsUsed']: j['extensionsUsed'].append('KHR_materials_unlit')
    p=P['HydrangeaCore']; C=g.get(p['attributes']['COLOR_0']); C[:,:3]=s2l((200,196,170)); g.put(p['attributes']['COLOR_0'],C)
    for nm in ['Leaf','LeafCard','Base']:
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); C[:,:3]*=LEAFMUL; g.put(p['attributes']['COLOR_0'],C)
    g.save(dst,{})
def bake_rows(T,shade):
    tl=s2l(T); return l2s(np.power(tl,1+SAT*(1-shade[:,None]))*np.power(shade[:,None],1.5))
if __name__=='__main__':
    run('web/FlowerBank_1m_sample_A_1002_v7.glb','FlowerBank_1m_sample_A.glb','web/FlowerBank_1m_sample_A_1002_v10.glb',float(sys.argv[1]))
