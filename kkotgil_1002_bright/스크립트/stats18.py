import json,struct,os,sys,numpy as np
def info(f):
    b=open(f,'rb').read(); l=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+l]); off=20+l+8
    tris=0; verts=0; dup=0; ds=[m.get('doubleSided',False) for m in j.get('materials',[])]
    CT={5123:np.uint16,5125:np.uint32,5121:np.uint8}
    for m in j['meshes']:
        for p in m['primitives']:
            verts+=j['accessors'][p['attributes']['POSITION']]['count']
            if 'indices' in p:
                a=j['accessors'][p['indices']]; bv=j['bufferViews'][a['bufferView']]; n=a['count']; tris+=n//3
                idx=np.frombuffer(b,dtype=CT[a['componentType']],count=n,offset=off+bv.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,3).astype(np.int64)
                s=set(map(tuple,np.sort(idx,1)[:,[0,1,2]].tolist())); 
                key=[tuple(sorted(t)) for t in idx.tolist()]
                from collections import Counter
                cnt=Counter(key); dup+=sum(1 for k,v in cnt.items() if v>=2)
            else: tris+=verts//3
    nimg=len(j.get('images',[]))
    return dict(tris=tris,verts=verts,size=os.path.getsize(f),dup=dup,ds=any(ds),img=nimg)
rows=[]
for f in sys.argv[1:]:
    i=info(f); back='있음(뒷면 삼각형 %d쌍, doubleSided=%s)'%(i['dup'],i['ds']) if i['dup']>0 else ('없음(doubleSided=%s)'%i['ds'])
    print(f"| {os.path.basename(f)} | {i['tris']:,} | {i['verts']:,} | {i['size']/1e6:.2f} MB | {i['img']} | {back} |")
