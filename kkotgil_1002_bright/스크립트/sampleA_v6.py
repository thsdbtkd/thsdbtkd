import sys,json,copy,numpy as np
from scipy.cluster.vq import kmeans2
from fbio import G
from sampleA_v5 import s2l,warm
HYD6=(226,234,214)
def run(src_v4,orig,dst,EMI,RS,HK,HEMI,seed=5):
    rng=np.random.default_rng(seed); g=G(src_v4); go=G(orig); j=g.j
    P={j['materials'][p['material']]['name']:p for p in j['meshes'][0]['primitives']}
    Po={go.j['materials'][p['material']]['name']:p for p in go.j['meshes'][0]['primitives']}
    # 수국: 기본색 (226,234,214) 안팎, 그늘 거의 없이(최저 0.85), 작은 무리마다 ±3
    p=P['Hydrangea_1']; X=g.get(p['attributes']['POSITION']); C=g.get(p['attributes']['COLOR_0'])
    Co=go.get(Po['Hydrangea_1']['attributes']['COLOR_0']); _,l=kmeans2(X,8,minit='++',seed=seed)
    lum=Co[:,:3]@[0.2126,0.7152,0.0722]; s=np.clip(lum/np.percentile(lum,80),0,1)
    pal=np.array([s2l(np.array(HYD6)+rng.normal(0,3,3)) for _ in range(8)])
    C[:,:3]=np.clip(pal[l]*HK*(0.85+0.15*s)[:,None],0,1); g.put(p['attributes']['COLOR_0'],C)
    fr,fb=warm((225,222,215)); lr,lb=warm((91,144,69))
    for nm,(r,b) in [('Flowers',(fr,fb)),('PetalEdge',(fr,fb)),('Hydrangea_1',(fr,fb)),('HydrangeaCore',(fr,fb)),('Leaf',(lr,lb)),('LeafCard',(lr,lb)),('Base',(lr,lb))]:
        p=P[nm]; C=g.get(p['attributes']['COLOR_0']); C[:,:3]*=(RS if nm in ('Flowers','PetalEdge') else 1.0); C[:,0]*=r; C[:,2]*=b; C[:,:3]=np.clip(C[:,:3],0,1); g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        if m['name'] in ('Flowers','PetalEdge'): m.pop('emissiveTexture',None) if m['name']=='Flowers' else None; m['emissiveFactor']=[EMI*1.02,EMI,EMI*0.97]; m['pbrMetallicRoughness']['roughnessFactor']=0.95
        if m['name']=='Hydrangea_1':
            m['occlusionTexture']['strength']=0.3          # 수국만 AO 영향 0.3
            m.pop('emissiveTexture',None)                  # 방사형(가운데 검정) 발광 텍스처 제거 → 고른 발광
            m['emissiveFactor']=[s2l(HYD6[0])*HEMI,s2l(HYD6[1])*HEMI,s2l(HYD6[2])*HEMI]
        if m['name']=='HydrangeaCore': m['emissiveFactor']=[s2l(HYD6[0])*HEMI*0.9,s2l(HYD6[1])*HEMI*0.9,s2l(HYD6[2])*HEMI*0.9]
    g.save(dst,{})
def mask(src,dst,targets):
    g=G(src); j=g.j
    for p in j['meshes'][0]['primitives']:
        nm=j['materials'][p['material']]['name']; C=g.get(p['attributes']['COLOR_0']); C[:,:3]=1.0 if nm in targets else 0.0; g.put(p['attributes']['COLOR_0'],C)
    for m in j['materials']:
        m.setdefault('extensions',{})['KHR_materials_unlit']={}
        m.pop('emissiveTexture',None); m.pop('emissiveFactor',None); m.pop('occlusionTexture',None); m.pop('normalTexture',None)
        if m['name'] in targets: m['pbrMetallicRoughness'].pop('baseColorTexture',None) if m['name'] in ('Flowers','HydrangeaCore') else None
    if 'KHR_materials_unlit' not in j.setdefault('extensionsUsed',[]): j['extensionsUsed'].append('KHR_materials_unlit')
    g.save(dst,{})
if __name__=='__main__':
    EMI,RS,HK,HEMI=map(float,sys.argv[1:5])
    run('web/FlowerBank_1m_sample_A_1002_v4.glb','FlowerBank_1m_sample_A.glb','web/FlowerBank_1m_sample_A_1002_v6.glb',EMI,RS,HK,HEMI)
    if len(sys.argv)>5:
        for src,tag in [('web/FlowerBank_1m_sample_A_1002_v6.glb','v6'),('FlowerBank_1m_sample_A.glb','now')]:
            mask(src,f'web/maskH_{tag}.glb',{'Hydrangea_1','HydrangeaCore'}); mask(src,f'web/maskR_{tag}.glb',{'Flowers','PetalEdge'})
