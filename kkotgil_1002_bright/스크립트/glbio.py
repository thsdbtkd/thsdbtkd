import json,struct,numpy as np
def load(f):
    b=open(f,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l]); off=20+l+8
    bv0=j['bufferViews'][0]; n=j['accessors'][0]['count']
    inter=np.frombuffer(b,dtype=np.float32,count=n*6,offset=off+bv0.get('byteOffset',0)).reshape(n,6).copy()
    bv1=j['bufferViews'][1]; m=j['accessors'][2]['count']
    idx=np.frombuffer(b,dtype=np.uint16,count=m,offset=off+bv1['byteOffset']).copy()
    return dict(raw=b,json=j,hlen=l,pos=inter[:,:3],col=inter[:,3:],idx=idx,off=off)
def save(d,col,f):
    b=bytearray(d['raw']); off=d['off']; n=len(col)
    inter=np.frombuffer(bytes(b[off:off+n*24]),dtype=np.float32).reshape(n,6).copy()
    inter[:,3:]=col.astype(np.float32)
    b[off:off+n*24]=inter.tobytes(); open(f,'wb').write(bytes(b))
