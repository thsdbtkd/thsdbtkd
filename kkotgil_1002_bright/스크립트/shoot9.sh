B="eye=1&unit=0.01&az=180"; J=""
for t in $@; do
 F=$([ $t = now ] && echo vr27n_D_bank.glb || echo vr27n_D_bank_bright_$t.glb); FM=$([ $t = now ] && echo vr27n_D_mound_c.glb || echo vr27n_D_mound_c_bright_$t.glb)
 for v in "eye|$B&dist=3.0&eyeh=1.5&pitch=12" "mid|$B&dist=2.0&eyeh=1.5&pitch=20" "close|$B&dist=1.0&eyeh=1.5&pitch=30"; do k=${v%%|*}; c=${v#*|}; J="$J shots/R_bank_${t}_$k.png|f=$F&$c shots/mF_bank_${t}_$k.png|f=maskF_vr27n_D_bank.glb&$c&bg=000000"; done
 for v in "mid|d=1.5" "close|close=1&cd=0.6&cx=-0.15&cy=0.65"; do k=${v%%|*}; c=${v#*|}; J="$J shots/R_mound_c_${t}_$k.png|f=$FM&$c shots/mF_mound_c_${t}_$k.png|f=maskF_vr27n_D_mound_c.glb&$c&bg=000000"; done
done
python3 cap.py $J >/dev/null
for t in $@; do for f in shots/R_*_${t}_*.png; do python3 measure9.py $f; done; done
