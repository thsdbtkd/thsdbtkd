# 샘플 A: 장면 + 수국/장미 마스크(검은 배경)
T=$1
V="mid|d=1.5 close|close=1&cd=0.6 153|close=1&cd=0.2&el=30&cx=-0.05&cy=0.6&cz=0.25"
J=""
for vv in $V; do k=${vv%%|*}; c=${vv#*|}
  if [ $T = now ]; then F=FlowerBank_1m_sample_A.glb; else F=FlowerBank_1m_sample_A_1002_v6.glb; fi
  J="$J shots/N_A_${T}_$k.png|f=$F&$c shots/mH_${T}_$k.png|f=maskH_$T.glb&$c&bg=000000 shots/mR_${T}_$k.png|f=maskR_$T.glb&$c&bg=000000"
done
python3 cap.py $J > /dev/null
for vv in $V; do k=${vv%%|*}; python3 measure6A.py "A_${T}_$k" shots/N_A_${T}_$k.png shots/mH_${T}_$k.png shots/mR_${T}_$k.png; done
