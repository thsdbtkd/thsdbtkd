import numpy as np
import recolor10 as R10, sampleA_v10 as SA
alt=np.random.default_rng(99); orig=R10.pick; rep=[0]
def pick(rng,pal):
    t=orig(rng,pal)
    if np.allclose(t,(232,108,142)) or np.allclose(t,(230,100,134)):
        t=np.array((242,156,178.) if alt.random()<0.5 else (248,196,184.)); rep[0]+=1
    return t
SA.pick=pick
SA.run('web/FlowerBank_1m_sample_A_1002_v7.glb','FlowerBank_1m_sample_A.glb','web/FlowerBank_1m_sample_A_1002_v13B.glb',0.55)
print('샘플 A 진분홍 송이 → 분홍·피치',rep[0])
