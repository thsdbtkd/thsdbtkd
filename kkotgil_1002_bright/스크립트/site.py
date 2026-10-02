"""현장 사진 위 합성: 고객 '현재 AR 화면' 사진에서 꽃이 없는 바닥 부분을 이어 붙여 배경으로, 렌더(배경색 키)를 얹음"""
import numpy as np,sys
from PIL import Image,ImageFilter
def site_bg(w,h):
    a=Image.open('ref3_left.png').convert('RGB'); W,H=a.size
    tile=a.crop((2,int(H*0.10),52,int(H*0.98)))   # 꽃 없는 왼쪽 바닥 띠(현장 사진)
    tw,th=tile.size; sc=max(w/(2*tw),h/(2*th)); tile=tile.resize((int(tw*sc)+1,int(th*sc)+1)); tw,th=tile.size
    out=Image.new('RGB',(w,h))
    for iy in range(0,h,th):
        for ix in range(0,w,tw):
            t=tile
            if (ix//tw)%2: t=t.transpose(Image.FLIP_LEFT_RIGHT)
            if (iy//th)%2: t=t.transpose(Image.FLIP_TOP_BOTTOM)
            out.paste(t,(ix,iy))
    return out
def comp(src,dst,key=(0x9a,0xa3,0x9f)):
    a=np.asarray(Image.open(src).convert('RGB')).astype(float); h,w=a.shape[:2]
    d=np.abs(a-np.array(key)).sum(-1); m=np.clip((d-6)/30,0,1)[...,None]
    bg=np.asarray(site_bg(w,h)).astype(float)
    Image.fromarray((a*m+bg*(1-m)).astype(np.uint8)).save(dst)
if __name__=='__main__': comp(sys.argv[1],sys.argv[2])
