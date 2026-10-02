B="eye=1&unit=0.01&az=180"; J=""
F=vr27n_D_bank_bright_v13A.glb; FM=vr27n_D_mound_c_bright_v13A.glb
for v in "eye|$B&dist=3.0&eyeh=1.5&pitch=12" "mid|$B&dist=2.0&eyeh=1.5&pitch=20" "close|$B&dist=1.0&eyeh=1.5&pitch=30"; do k=${v%%|*}; c=${v#*|}; J="$J shots/V_bank_v13A_$k.png|f=$F&$c shots/mH13_bank_$k.png|f=maskH13_vr27n_D_bank.glb&$c&bg=000000"; done
for v in "mid|d=1.5" "close|close=1&cd=0.6&cx=-0.15&cy=0.65"; do k=${v%%|*}; c=${v#*|}; J="$J shots/V_mound_c_v13A_$k.png|f=$FM&$c shots/mH13_mound_c_$k.png|f=maskH13_vr27n_D_mound_c.glb&$c&bg=000000"; done
python3 cap.py $J >/dev/null
python3 - <<'P'
import numpy as np,glob
from PIL import Image
from scipy.ndimage import binary_erosion
BG=np.array([154,163,159])
for f in sorted(glob.glob('shots/V_*_v13A_*.png')):
    a=np.asarray(Image.open(f).convert('RGB')).astype(float); mk=f.replace('V_','mH13_').replace('_v13A','')
    m=binary_erosion(np.asarray(Image.open(mk).convert('L'))>100,iterations=1); h=a[m]
    br=h.mean(1); sat=(h.max(1)-h.min(1))/np.maximum(h.max(1),1); gray=(br<0.7*255)&(sat<0.12)
    obj=np.abs(a-BG).sum(-1)>18; green=obj&(a[...,1]>a[...,0]+8)&(a[...,1]>=a[...,2]); lf=a[green]
    print(f"{f.split('/')[-1]}: 수국 화소 {m.sum()} 회색(밝기<178·무채색) {gray.mean()*100:.1f}% 수국 평균 {br.mean():.0f} | 잎 중앙 밝기 {np.median(lf.mean(1)):.0f} 중앙 RGB {tuple(np.median(lf,0).round(0).astype(int).tolist())}")
P
