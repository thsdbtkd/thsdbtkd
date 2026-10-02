import json,struct,io,numpy as np
from PIL import Image
def lift(img,strength):
    a=np.asarray(img.convert('RGBA')).astype(float)/255; rgb=a[...,:3]
    L=rgb@np.array([0.299,0.587,0.114]); Lt=L**strength
    g=np.clip(Lt/np.maximum(L,1e-3),1,1.9)[...,None]
    out=np.clip(rgb*g,0,240/255)
    a[...,:3]=np.minimum(out,np.maximum(rgb,out)); return a
def run(src,dst,strength=0.72):
    b=open(src,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l]); off=20+l+8
    binlen=struct.unpack('<I',b[20+l:24+l])[0]; BIN=b[off:off+binlen]
    imgviews={im['bufferView']:im for im in j['images']}
    new=bytearray()
    for i,bv in enumerate(j['bufferViews']):
        data=BIN[bv.get('byteOffset',0):bv.get('byteOffset',0)+bv['byteLength']]
        if i in imgviews:
            im=imgviews[i]; img=Image.open(io.BytesIO(data)); a=lift(img,strength)
            if im['mimeType'].endswith('png'):
                o=Image.fromarray((a*255).round().astype(np.uint8),'RGBA'); buf=io.BytesIO(); o.save(buf,'PNG',optimize=True)
            else:
                o=Image.fromarray((a[...,:3]*255).round().astype(np.uint8),'RGB'); buf=io.BytesIO(); o.save(buf,'JPEG',quality=90)
            data=buf.getvalue()
        while len(new)%4: new.append(0)
        bv['byteOffset']=len(new); bv['byteLength']=len(data); new+=data
    while len(new)%4: new.append(0)
    j['buffers'][0]['byteLength']=len(new)
    js=json.dumps(j,separators=(',',':')).encode(); js+=b' '*((4-len(js)%4)%4)
    out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(new))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(new),0x004E4942)+bytes(new)
    open(dst,'wb').write(out)
    print(dst,len(out))
if __name__=='__main__': run('mound_a.glb','web/mound_a_AOlight.glb',0.6)
