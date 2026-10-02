F=FlowerBank_1m_sample_A_1002_v7.glb
python3 cap.py "shots/P_A_v7_mid.png|f=$F&d=1.5" "shots/P_A_v7_close.png|f=$F&close=1&cd=0.6" "shots/P_A_v7_153.png|f=$F&close=1&cd=0.2&el=30&cx=-0.05&cy=0.6&cz=0.25" >/dev/null
python3 measure7.py shots/P_A_v7_*.png
