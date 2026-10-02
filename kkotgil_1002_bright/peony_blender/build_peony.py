"""Blender(bpy)에서 고품질 작약 송이 만들기: 겹꽃잎 ~90장(꽃잎당 9x12 격자, 컵·주름·끝 말림), 꽃잎 텍스처(알파 윤곽),
Cycles 로 AO 를 정점색에 구움 → glb 내보내기. 형태 변형 4종."""
import bpy, bmesh, math, random, sys, numpy as np
from mathutils import Vector, Matrix
NU,NV=9,12
def petal_mesh(length,width,tilt,cup,curl,ruf,seed):
    r=random.Random(seed); ph=[r.uniform(0,6.28) for _ in range(3)]
    verts=[];uvs=[]
    for i in range(NV):
        v=i/(NV-1)
        w=width*math.sin(math.pi*(0.06+0.58*v))**0.6*(0.55+0.45*v)*(1-0.25*max(0,v-0.85)/0.15)
        for k in range(NU):
            uu=k/(NU-1)-0.5
            ang=tilt+curl*v**1.6
            cupz=cup*(uu*2)**2*w*0.55
            rz=ruf*length*v**2*(0.6*math.sin(uu*9+ph[0])+0.4*math.sin(uu*17+ph[1]))
            x=v*length*math.cos(ang)-cupz*math.sin(ang)
            z=v*length*math.sin(ang)+cupz*math.cos(ang)+rz
            y=uu*w*2*1.08
            verts.append((x,y,z)); uvs.append((0.5+uu*1.0*(w/max(width,1e-6))*0.5/0.5*0.5+0.0, v))
    faces=[]
    for i in range(NV-1):
        for k in range(NU-1):
            a=i*NU+k; faces.append((a,a+NU,a+NU+1,a+1))
    # UV: 꽃잎 텍스처 전체(u 0~1 가로, v 0~1 세로)
    uvs=[(k/(NU-1), i/(NV-1)) for i in range(NV) for k in range(NU)]
    return verts,faces,uvs
RINGS=[(10,1.00,0.10,0.30,0.35,0.05,0.00),(10,0.92,0.32,0.50,0.45,0.05,0.06),(11,0.82,0.62,0.65,0.50,0.05,0.16),
       (11,0.72,0.90,0.75,0.50,0.06,0.28),(12,0.62,1.10,0.85,0.45,0.07,0.42),(12,0.52,1.25,0.90,0.40,0.08,0.56),
       (14,0.42,1.38,0.95,0.30,0.09,0.70),(14,0.32,1.48,1.0,0.20,0.10,0.82),(10,0.22,1.52,1.0,0.10,0.10,0.92)]
def build(variant,seed):
    r=random.Random(seed); R=1.0
    V=[];F=[];UV=[];D=[]
    for ri,(n,lf,tilt,cup,curl,ruf,dep) in enumerate(RINGS):
        for j in range(n):
            phi=2*math.pi*j/n+ri*0.43+r.gauss(0,0.12)
            length=R*lf*r.uniform(0.88,1.08); width=length*r.uniform(0.48,0.62)*(1.25 if ri<2 else 1.0)
            vs,fs,uv=petal_mesh(length,width,tilt+r.gauss(0,0.08)+variant*0.03,cup,curl,ruf,seed*1000+ri*50+j)
            c,s=math.cos(phi),math.sin(phi); off=V.__len__()
            for (x,y,z) in vs:
                x2=x+R*0.05*(1-lf); z2=z+R*0.10*dep
                V.append((x2*c-y*s, x2*s+y*c, z2)); D.append(dep)
            F+=[tuple(a+off for a in f) for f in fs]; UV+=uv
    me=bpy.data.meshes.new(f'peony{variant}'); me.from_pydata(V,[],F); me.update()
    uvl=me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        for li in poly.loop_indices: uvl.data[li].uv=UV[me.loops[li].vertex_index]
    ob=bpy.data.objects.new(f'peony{variant}',me); bpy.context.collection.objects.link(ob)
    for p in me.polygons: p.use_smooth=True
    # 안쪽 깊이(dep)를 정점색 G 채널에 저장(나중에 색 그러데이션용) — AO 는 R 에
    ca=me.color_attributes.new('Col','FLOAT_COLOR','POINT')
    for i,d in enumerate(D): ca.data[i].color=(1,d,0,1)
    return ob
bpy.ops.wm.read_factory_settings(use_empty=True)
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=128; sc.cycles.device='CPU'
w=bpy.data.worlds.new('w'); sc.world=w; w.light_settings.distance=0.12
for vi in range(4):
    for o in list(bpy.data.objects):
        if o.name!='Sun': bpy.data.objects.remove(o,do_unlink=True)
    ob=build(vi,11+vi*7)
    # AO 굽기용 재질(정점색 대상)
    mat=bpy.data.materials.new('m'); mat.use_nodes=True; ob.data.materials.append(mat)
    bpy.context.view_layer.objects.active=ob; ob.select_set(True)
    ao=ob.data.color_attributes.new('AO','FLOAT_COLOR','POINT'); ob.data.color_attributes.active_color=ao; ob.data.color_attributes.render_color_index=ob.data.color_attributes.active_color_index
    sc.render.bake.target='VERTEX_COLORS'; sc.cycles.bake_type='AO'
    bpy.ops.object.bake(type='AO',target='VERTEX_COLORS')
    import numpy as np
    a=np.array([c.color[0] for c in ob.data.color_attributes['AO'].data]); a2=np.array([c.color[0] for c in ob.data.color_attributes['Col'].data]); print('AO attr',a.mean(),'Col attr',a2.mean()); d=np.array([c.color[1] for c in ob.data.color_attributes['Col'].data])
    np.save(f'peony{vi}_ao.npy',a); np.save(f'peony{vi}_dep.npy',d)
    # 빛 굽기: 따뜻한 해(왼쪽 위) + 밝은 하늘, 꽃잎 반투명 → Diffuse(직접+간접, 색 제외)를 정점색에
    nt=mat.node_tree; bsdf=nt.nodes['Principled BSDF']; bsdf.inputs['Base Color'].default_value=(0.9,0.85,0.83,1)
    bsdf.inputs['Roughness'].default_value=0.6
    try: bsdf.inputs['Subsurface Weight'].default_value=0.25
    except Exception: pass
    try: bsdf.inputs['Transmission Weight'].default_value=0.15
    except Exception: pass
    w.use_nodes=True; bg=w.node_tree.nodes['Background']; bg.inputs[0].default_value=(1.0,0.93,0.85,1); bg.inputs[1].default_value=0.9
    if 'Sun' not in bpy.data.objects:
        ld=bpy.data.lights.new('Sun','SUN'); ld.energy=3.2; ld.color=(1.0,0.90,0.78); ld.angle=0.25
        so=bpy.data.objects.new('Sun',ld); bpy.context.collection.objects.link(so)
        so.rotation_euler=(math.radians(40),math.radians(-30),math.radians(20))
    li=ob.data.color_attributes.new('Light','FLOAT_COLOR','POINT'); ob.data.color_attributes.active_color=li; ob.data.color_attributes.render_color_index=ob.data.color_attributes.active_color_index
    sc.render.bake.use_pass_direct=True; sc.render.bake.use_pass_indirect=True; sc.render.bake.use_pass_color=False
    bpy.ops.object.bake(type='DIFFUSE',target='VERTEX_COLORS',pass_filter={'DIRECT','INDIRECT'})
    lt=np.array([c.color[:3] for c in ob.data.color_attributes['Light'].data]); np.save(f'peony{vi}_light.npy',lt); print('light',lt.mean(0).round(3),lt.max().round(2))
    me=ob.data; vs=np.array([v.co[:] for v in me.vertices]); np.save(f'peony{vi}_v.npy',vs)
    nm=np.array([v.normal[:] for v in me.vertices]); np.save(f'peony{vi}_n.npy',nm)
    tri=[];uv=np.zeros((len(vs),2))
    for p in me.polygons:
        l=list(p.vertices); tri+=[[l[0],l[1],l[2]],[l[0],l[2],l[3]]]
        for li in p.loop_indices: uv[me.loops[li].vertex_index]=me.uv_layers[0].data[li].uv
    np.save(f'peony{vi}_t.npy',np.array(tri)); np.save(f'peony{vi}_uv.npy',uv)
    print('peony',vi,'verts',len(vs),'tris',len(tri),'AO',a.min().round(2),a.mean().round(2))
    ob.select_set(False)
