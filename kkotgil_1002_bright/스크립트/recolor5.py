import sys,json,struct,numpy as np
import recolor5_base as R
from glbio import load
def s2l(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,12.92*x,1.055*x**(1/2.4)-0.055)*255
def write_backface(d,col,out,extra=None):
    """뒷면 삼각형 복제(감김 반전), 정점색 그대로 공유, doubleSided=false"""
    j=json.loads(json.dumps(d['json'])); n=len(col)
    inter=np.hstack([d['pos'],col]).astype(np.float32)
    t=d['idx'].reshape(-1,3)
    if extra is not None and len(extra): t=np.vstack([t,extra.astype(t.dtype)])
    idx=np.vstack([t,t[:,[0,2,1]]]).astype(np.uint16).ravel()
    vb=inter.tobytes(); ib=idx.tobytes()
    pad=lambda b:b+b'\0'*((4-len(b)%4)%4)
    binb=pad(vb)+pad(ib)
    j['bufferViews'][0].update(byteOffset=0,byteLength=len(vb))
    j['bufferViews'][1].update(byteOffset=len(pad(vb)),byteLength=len(ib))
    j['accessors'][2]['count']=len(idx); j['buffers'][0]['byteLength']=len(binb)
    for m in j['materials']: m['doubleSided']=False
    js=json.dumps(j,separators=(',',':')).encode(); js+=b' '*((4-len(js)%4)%4)
    open(out,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb)
def run(src,seed,AO,CREV,PN,WARM=4):
    col,flower=R.run(src+'.glb',None,seed,AO=AO,CREV=CREV,TOP=242,PN=PN,ret=True)
    v4=load('web/'+src+'_bright_v4.glb'); leaf=~flower
    col[leaf]=v4['col'][leaf]                     # 4판 잎 톤 유지
    sg=l2s(col); sg[:,0]+=WARM; sg[:,2]-=WARM     # 따뜻한 햇살: R+4, B-4
    sg[flower]=np.minimum(sg[flower],242); sg[leaf]=np.minimum(sg[leaf],220)
    col=s2l(np.clip(sg,0,255)); d=load(src+'.glb')
    from holes2 import band_holes
    from seg import seg
    S=seg(src+'.glb'); lf=((S['ch']>=60)&(S['ch']<170))[S['lab']]
    extra,info=band_holes(d,lf); print(src,'띠 구멍 채움',len(info),'칸')
    write_backface(d,col,'web/'+src+'_bright_v5.glb',extra)
if __name__=='__main__':
    AO,CREV,PN=map(float,sys.argv[1:4]); R.WGAIN=float(sys.argv[4]) if len(sys.argv)>4 else 1.0
    run('vr27n_D_bank',7,AO,CREV,PN); run('vr27n_D_mound_c',11,AO,CREV,PN)
