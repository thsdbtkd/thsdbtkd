import json,struct,numpy as np
CT={5126:np.float32,5121:np.uint8,5123:np.uint16,5125:np.uint32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
class G:
    def __init__(s,f):
        b=open(f,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; s.j=json.loads(b[20:20+l]); off=20+l+8
        bl=struct.unpack('<I',b[20+l:24+l])[0]; s.bin=bytearray(b[off:off+bl])
    def view(s,ai):
        a=s.j['accessors'][ai]; bv=s.j['bufferViews'][a['bufferView']]; dt=np.dtype(CT[a['componentType']]); n=NC[a['type']]
        st=bv.get('byteStride') or dt.itemsize*n; o=bv.get('byteOffset',0)+a.get('byteOffset',0)
        return a,dt,n,st,o
    def get(s,ai):
        a,dt,n,st,o=s.view(ai); c=a['count']
        raw=np.frombuffer(bytes(s.bin[o:o+st*(c-1)+dt.itemsize*n]),dtype=np.uint8)
        idx=(np.arange(c)[:,None]*st+np.arange(dt.itemsize*n)[None,:])
        arr=raw[idx].copy().view(dt).reshape(c,n).astype(np.float64)
        if a.get('normalized'): arr/=np.iinfo(dt).max
        return arr
    def put(s,ai,arr):
        a,dt,n,st,o=s.view(ai); c=a['count']
        if a.get('normalized'): arr=np.round(np.clip(arr,0,1)*np.iinfo(dt).max)
        by=np.ascontiguousarray(arr.astype(dt)).view(np.uint8).reshape(c,-1)
        for k in range(c): s.bin[o+k*st:o+k*st+by.shape[1]]=by[k].tobytes()
    def imgbytes(s,i):
        bv=s.j['bufferViews'][s.j['images'][i]['bufferView']]; o=bv.get('byteOffset',0); return bytes(s.bin[o:o+bv['byteLength']])
    def save(s,f,newimgs={}):
        j=s.j; iv={im['bufferView']:k for k,im in enumerate(j['images'])}; new=bytearray()
        for i,bv in enumerate(j['bufferViews']):
            o=bv.get('byteOffset',0); data=bytes(s.bin[o:o+bv['byteLength']])
            if i in iv and iv[i] in newimgs: data,mt=newimgs[iv[i]]; j['images'][iv[i]]['mimeType']=mt
            while len(new)%4: new.append(0)
            bv['byteOffset']=len(new); bv['byteLength']=len(data); new+=data
        while len(new)%4: new.append(0)
        j['buffers'][0]['byteLength']=len(new)
        js=json.dumps(j,separators=(',',':')).encode(); js+=b' '*((4-len(js)%4)%4)
        open(f,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(new))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(new),0x004E4942)+bytes(new))
