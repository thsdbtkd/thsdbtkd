"""서울 지하철 섬식 승강장 한 구간 예시 (일반형, 가상 역명). 단위 m, Z-up.
X = 승강장 길이 방향(0~39, 19.5m 차량 2량분), Y = 폭 방향(-5~5, 섬식 10m), Z = 높이(바닥 0, 천장 3.3).
근거: 조사_승강장예시_20261001.md
사용: python3 build.py <cam> <samples> <res%> [clay]   cam = c1|c2|c3|c4
"""
import bpy, math, sys, os
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
CAM = args[0] if len(args) > 0 else 'c1'
SAMPLES = int(args[1]) if len(args) > 1 else 32
RES = int(args[2]) if len(args) > 2 else 50
CLAY = len(args) > 3 and args[3] == 'clay'
OUT = os.path.dirname(os.path.abspath(__file__))

L = 39.0          # 길이
HW = 5.0          # 반폭
CH = 3.3          # 천장고
PH = 1.14         # 승강장 높이(레일면 위)
PITCH = 19.5 / 4  # 문 간격 4.875
LINE = (0.0, 0.62, 0.29, 1)  # 노선색(녹색 계열, 가상)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
col = sc.collection

# ---------------- 재질 ----------------
def node_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    return m, nt, b

def simple(name, color, rough=0.5, metal=0.0, emit=None, estr=0.0, alpha=1.0, trans=0.0):
    m, nt, b = node_mat(name)
    b.inputs['Base Color'].default_value = color
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if emit:
        b.inputs['Emission Color'].default_value = emit
        b.inputs['Emission Strength'].default_value = estr
    if trans:
        b.inputs['Transmission Weight'].default_value = trans
        b.inputs['IOR'].default_value = 1.5
    if alpha < 1:
        b.inputs['Alpha'].default_value = alpha
    return m

def grime(nt, b, base_rough, amount=0.12, scale=3.0):
    """거칠기에 얼룩(노이즈)을 섞어 너무 깨끗한 CG 티를 줄인다."""
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = 8
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = base_rough
    mr.inputs['To Max'].default_value = base_rough + amount
    nt.links.new(n.outputs['Fac'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], b.inputs['Roughness'])
    return tc

def floor_mat():
    m, nt, b = node_mat('바닥석재')
    tc = grime(nt, b, 0.05, 0.12, 1.2)
    br = nt.nodes.new('ShaderNodeTexBrick')
    br.offset = 0.0
    br.inputs['Scale'].default_value = 1.0
    br.inputs['Mortar Size'].default_value = 0.004
    br.inputs['Brick Width'].default_value = 0.6
    br.inputs['Row Height'].default_value = 0.6
    br.inputs['Color1'].default_value = (0.56, 0.55, 0.53, 1)
    br.inputs['Color2'].default_value = (0.50, 0.50, 0.49, 1)
    br.inputs['Bias'].default_value = 0.0
    br.inputs['Mortar'].default_value = (0.42, 0.42, 0.41, 1)
    nt.links.new(tc.outputs['Object'], br.inputs['Vector'])
    # 화강석 점박이
    v = nt.nodes.new('ShaderNodeTexVoronoi'); v.inputs['Scale'].default_value = 140
    nt.links.new(tc.outputs['Object'], v.inputs['Vector'])
    cr = nt.nodes.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position = 0.0; cr.color_ramp.elements[0].color = (0.25, 0.25, 0.26, 1)
    cr.color_ramp.elements[1].position = 0.35; cr.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(v.outputs['Distance'], cr.inputs['Fac'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = 0.7
    nt.links.new(br.outputs['Color'], mix.inputs['A'])
    nt.links.new(cr.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    bm = nt.nodes.new('ShaderNodeBump'); bm.inputs['Strength'].default_value = 0.12
    inv = nt.nodes.new('ShaderNodeMath'); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
    nt.links.new(br.outputs['Fac'], inv.inputs[1])
    nt.links.new(inv.outputs['Value'], bm.inputs['Height'])
    nt.links.new(bm.outputs['Normal'], b.inputs['Normal'])
    return m

def panel_mat(name, c1, c2, w, h, rough=0.35):
    m, nt, b = node_mat(name)
    tc = grime(nt, b, rough, 0.1, 2.0)
    br = nt.nodes.new('ShaderNodeTexBrick'); br.offset = 0.0
    br.inputs['Mortar Size'].default_value = 0.006
    br.inputs['Brick Width'].default_value = w
    br.inputs['Row Height'].default_value = h
    br.inputs['Color1'].default_value = c1; br.inputs['Color2'].default_value = c2
    br.inputs['Mortar'].default_value = (0.35, 0.36, 0.37, 1)
    nt.links.new(tc.outputs['Object'], br.inputs['Vector'])
    nt.links.new(br.outputs['Color'], b.inputs['Base Color'])
    return m

def tactile_mat():
    m, nt, b = node_mat('점자블록')
    b.inputs['Base Color'].default_value = (0.85, 0.62, 0.02, 1)
    tc = grime(nt, b, 0.55, 0.15, 4)
    # 선형 돌기
    w = nt.nodes.new('ShaderNodeTexWave'); w.wave_type = 'BANDS'; w.bands_direction = 'X'
    w.inputs['Scale'].default_value = 3.0
    nt.links.new(tc.outputs['Object'], w.inputs['Vector'])
    bm = nt.nodes.new('ShaderNodeBump'); bm.inputs['Strength'].default_value = 0.6
    nt.links.new(w.outputs['Fac'], bm.inputs['Height'])
    nt.links.new(bm.outputs['Normal'], b.inputs['Normal'])
    return m

def brushed_steel():
    m, nt, b = node_mat('스테인리스')
    b.inputs['Base Color'].default_value = (0.72, 0.73, 0.75, 1)
    b.inputs['Metallic'].default_value = 1.0
    grime(nt, b, 0.30, 0.08, 1.5)
    return m

M = {}
if CLAY:
    clay = simple('클레이', (0.78, 0.78, 0.77, 1), 0.7)
    for k in ['floor', 'ceil', 'wall', 'tact', 'steel', 'psdframe', 'header', 'line', 'glass', 'dark',
              'track', 'rail', 'red', 'navy', 'white', 'tunnel', 'bench', 'esc', 'rubber']:
        M[k] = clay
    M['glass'] = simple('클레이유리', (0.9, 0.9, 0.9, 1), 0.05, trans=1.0)
else:
    M['floor'] = floor_mat()
    M['ceil'] = panel_mat('천장판', (0.88, 0.88, 0.87, 1), (0.86, 0.86, 0.85, 1), 1.2, 0.6, 0.6)
    M['wall'] = panel_mat('벽패널', (0.82, 0.82, 0.80, 1), (0.79, 0.79, 0.78, 1), 1.2, 0.6, 0.25)
    M['tact'] = tactile_mat()
    M['steel'] = brushed_steel()
    M['psdframe'] = simple('PSD프레임', (0.42, 0.44, 0.46, 1), 0.35, 0.8)
    M['header'] = simple('PSD머리판', (0.30, 0.31, 0.33, 1), 0.4, 0.3)
    M['line'] = simple('노선색', LINE, 0.4)
    M['glass'] = simple('유리', (0.95, 0.97, 0.97, 1), 0.02, trans=1.0)
    M['dark'] = simple('검정', (0.02, 0.02, 0.02, 1), 0.5)
    M['track'] = panel_mat('도상', (0.08, 0.08, 0.08, 1), (0.07, 0.07, 0.07, 1), 0.6, 2.5, 0.8)
    M['rail'] = simple('레일', (0.35, 0.33, 0.30, 1), 0.3, 1.0)
    M['red'] = simple('소화기함', (0.62, 0.04, 0.03, 1), 0.35)
    M['navy'] = simple('안내판', (0.04, 0.07, 0.16, 1), 0.4)
    M['white'] = simple('흰색', (0.85, 0.85, 0.85, 1), 0.4)
    M['tunnel'] = panel_mat('선로벽', (0.55, 0.56, 0.56, 1), (0.50, 0.51, 0.52, 1), 2.4, 1.2, 0.6)
    M['bench'] = brushed_steel()
    M['esc'] = simple('에스컬레이터', (0.45, 0.46, 0.47, 1), 0.35, 0.9)
    M['rubber'] = simple('손잡이고무', (0.02, 0.02, 0.02, 1), 0.5)
M['led'] = simple('LED', (1, 1, 1, 1), 0.5, emit=(1, 0.98, 0.95, 1), estr=8.0)
M['sign_txt'] = simple('글자흰', (1, 1, 1, 1), 0.5, emit=(1, 1, 1, 1), estr=2.5)
M['sign_yel'] = simple('글자노랑', (1, 0.8, 0.1, 1), 0.5, emit=(1, 0.8, 0.1, 1), estr=2.5)
def ad_mat():
    m, nt, b = node_mat('광고등')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 0.6; n.inputs['Detail'].default_value = 1
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    cr = nt.nodes.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].color = (0.05, 0.35, 0.75, 1); cr.color_ramp.elements[1].color = (0.95, 0.55, 0.15, 1)
    nt.links.new(n.outputs['Fac'], cr.inputs['Fac'])
    nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
    nt.links.new(cr.outputs['Color'], b.inputs['Emission Color'])
    b.inputs['Emission Strength'].default_value = 1.6
    return m
M['ad'] = ad_mat()

# ---------------- 도우미 ----------------
def box(name, cx, cy, cz, sx, sy, sz, mat, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    o = bpy.context.active_object; o.name = name
    o.scale = (sx, sy, sz); o.rotation_euler[2] = rot_z
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    return o

def cyl(name, cx, cy, cz, r, h, mat, verts=48, axis='Z'):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=(cx, cy, cz))
    o = bpy.context.active_object; o.name = name
    if axis == 'X': o.rotation_euler[1] = math.pi / 2
    if axis == 'Y': o.rotation_euler[0] = math.pi / 2
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    return o

def text(s, x, y, z, size, mat, rot=(math.pi / 2, 0, 0), align='CENTER'):
    bpy.ops.object.text_add(location=(x, y, z), rotation=rot)
    o = bpy.context.active_object
    o.data.body = s
    try:
        o.data.font = bpy.data.fonts.load(FONT)
    except Exception:
        pass
    o.data.size = size; o.data.align_x = align; o.data.align_y = 'CENTER'
    o.data.extrude = 0.002
    o.data.materials.append(mat)
    return o

def area_light(x, y, z, sx, sy, power, temp=(1, 0.97, 0.92)):
    bpy.ops.object.light_add(type='AREA', location=(x, y, z))
    l = bpy.context.active_object; l.data.shape = 'RECTANGLE'
    l.data.size = sx; l.data.size_y = sy; l.data.energy = power; l.data.color = temp
    l.rotation_euler = (math.pi, 0, 0)
    return l

# ---------------- 구조 ----------------
X0, X1 = -6.0, L + 6.0   # 화면 밖까지 조금 더
LX = X1 - X0; MX = (X0 + X1) / 2
# 승강장 바닥
box('승강장바닥', MX, 0, -0.1, LX, 2 * HW, 0.2, M['floor'])
# 천장판(계단 개구부 제외: x 14~26.5, y -2.2~2.2)
SX0, SX1, SY = 14.0, 26.5, 2.25
for (a, b_) in [(X0, SX0), (SX1, X1)]:
    box('천장', (a + b_) / 2, 0, CH + 0.02, b_ - a, 2 * HW, 0.04, M['ceil'])
box('천장좌', (SX0 + SX1) / 2, -(HW + SY) / 2, CH + 0.02, SX1 - SX0, HW - SY, 0.04, M['ceil'])
box('천장우', (SX0 + SX1) / 2, (HW + SY) / 2, CH + 0.02, SX1 - SX0, HW - SY, 0.04, M['ceil'])
# 천장 위 어두운 덮개(빛 새는 것 방지)
box('천장위', MX, 0, CH + 0.8, LX, 2 * HW + 10, 0.1, M['dark'])

# LED 사각 조명 2줄 + 실제 광원
for i in range(int(LX / 2.4) + 1):
    x = X0 + 1.2 + i * 2.4
    for y in (-2.6, 2.6):
        if SX0 - 0.3 < x < SX1 + 0.3 and abs(y) < SY + 0.4:
            continue
        box('LED', x, y, CH - 0.005, 1.2, 0.3, 0.01, M['led'])
        area_light(x, y, CH - 0.03, 1.2, 0.3, 55)
# 스크린도어 쪽 라인 조명
for y in (-(HW - 0.45), HW - 0.45):
    box('라인LED', MX, y, CH - 0.005, LX, 0.08, 0.01, M['led'])

# ---------------- 스크린도어(밀폐형) ----------------
PSD_TOP = 2.15
def psd_side(sgn):
    y = sgn * HW
    # 문틀 기준선: 차량 문 중심 = 1.22 + k*PITCH (차량 시작 0)
    centers = []
    for car in range(-1, 4):
        for k in range(4):
            c = car * 19.5 + 1.22 + k * PITCH + 0.6
            if X0 < c < X1: centers.append(c)
    # 하부 문지방(스테인리스), 머리판
    box('문지방', MX, y, 0.03, LX, 0.25, 0.06, M['steel'])
    box('머리판', MX, y, (PSD_TOP + CH) / 2, LX, 0.22, CH - PSD_TOP, M['header'])
    box('노선띠', MX, y - sgn * 0.115, PSD_TOP + 0.18, LX, 0.01, 0.07, M['line'])
    # 고정 유리 패널(문 사이 전체 길이 유리, 문 위치에서 끊음)
    edges = [X0]
    for c in centers:
        edges += [c - 1.0, c + 1.0]
    edges.append(X1)
    for i in range(0, len(edges), 2):
        a, b_ = edges[i], edges[i + 1]
        if b_ - a < 0.1: continue
        box('고정유리', (a + b_) / 2, y, PSD_TOP / 2 + 0.03, b_ - a, 0.02, PSD_TOP - 0.06, M['glass'])
        # 고정부 중간 기둥틀
        n = max(1, int((b_ - a) / 1.0))
        for j in range(n + 1):
            box('틀', a + j * (b_ - a) / n, y, PSD_TOP / 2, 0.06, 0.12, PSD_TOP, M['psdframe'])
    for c in centers:
        # 미닫이 문 2짝(닫힘), 틀
        for s in (-1, 1):
            box('문유리', c + s * 0.5, y, PSD_TOP / 2 + 0.05, 0.94, 0.03, PSD_TOP - 0.25, M['glass'])
            box('문틀세로', c + s * 0.97, y, PSD_TOP / 2, 0.06, 0.1, PSD_TOP, M['psdframe'])
        box('문틀하', c, y, 0.12, 2.0, 0.06, 0.14, M['psdframe'])
        box('문맞닿음', c, y, PSD_TOP / 2, 0.04, 0.06, PSD_TOP - 0.2, M['rubber'])
        # 문 위 표시창(칸-문 번호)
        box('표시창', c, y - sgn * 0.12, PSD_TOP + 0.45, 0.6, 0.02, 0.22, M['dark'])
        text(f'{centers.index(c) % 4 + 1}-{centers.index(c) // 4 + 1}', c, y - sgn * 0.135, PSD_TOP + 0.45, 0.13,
             M['sign_yel'], rot=(math.pi / 2, 0, 0 if sgn > 0 else math.pi))
        # 바닥 승차 위치 표시(노란 화살표 대신 노란 띠 2줄)
        for s in (-1, 1):
            box('승차표시', c + s * 0.7, y - sgn * 0.45, 0.002, 0.5, 0.08, 0.004, M['tact'])
    return centers

CENTERS = psd_side(1)
psd_side(-1)

# 점자블록(스크린도어에서 0.6m 띄워 폭 0.3)
for sgn in (-1, 1):
    box('점자선', MX, sgn * (HW - 0.75), 0.004, LX, 0.3, 0.008, M['tact'])

# ---------------- 선로·터널(유리 너머) ----------------
for sgn in (-1, 1):
    yt = sgn * (HW + 1.75)
    box('선로바닥', MX, yt, -PH - 0.1, LX, 3.6, 0.2, M['track'])
    for g in (-0.7175, 0.7175):
        box('레일', MX, yt + g, -PH + 0.08, LX, 0.07, 0.16, M['rail'])
    box('선로벽', MX, sgn * (HW + 3.7), (CH - PH) / 2 - 0.2, LX, 0.2, CH + PH + 0.4, M['tunnel'])
    box('승강장측벽', MX, sgn * (HW + 0.12), -PH / 2, LX, 0.25, PH, M['tunnel'])
    # 선로 위 조명(유리 너머가 실제처럼 밝게 보이도록)
    for i in range(int(LX / 4)):
        area_light(X0 + 2 + i * 4, yt, CH - 0.1, 1.2, 0.4, 45)
        box('선로LED', X0 + 2 + i * 4, yt, CH - 0.06, 1.2, 0.15, 0.01, M['led'])
    box('선로천장', MX, yt, CH, LX, 3.8, 0.05, M['ceil'])
    # 선로벽 광고 조명판
    for i in range(6):
        x = X0 + 4 + i * 8
        box('광고', x, sgn * (HW + 3.58), 1.0, 3.0, 0.04, 1.6, M['ad'])
        box('광고틀', x, sgn * (HW + 3.6), 1.0, 3.2, 0.03, 1.8, M['steel'])

# ---------------- 기둥(원형 스테인리스, 연단에서 1.5m 이상) ----------------
COL_Y = 2.75
for i in range(8):
    x = X0 + 2.5 + i * 6.5
    for sgn in (-1, 1):
        if SX0 - 0.5 < x < SX1 + 0.5: continue
        cyl('기둥', x, sgn * COL_Y, CH / 2, 0.4, CH, M['steel'])
        cyl('기둥받침', x, sgn * COL_Y, 0.06, 0.43, 0.12, M['dark'])
        # 소화기함(일부)
        if i % 2 == 0:
            box('소화기함', x, sgn * (COL_Y - 0.42), 0.75, 0.36, 0.14, 0.6, M['red'])

# ---------------- 계단 + 에스컬레이터(승강장 가운데, +x 방향 상행) ----------------
RISE, RUN = 0.165, 0.33
N = 18                     # 약 3m 올라가 천장 위로 사라짐
STW = 3.0                  # 계단 폭
ESW = 1.2                  # 에스컬레이터 폭
YS0 = -SY + 0.05           # 계단 시작 y
for k in range(N):
    x = SX0 + 0.6 + k * RUN
    z = (k + 1) * RISE
    box('디딤', x + RUN / 2, YS0 + STW / 2, z - RISE / 2, RUN, STW, RISE, M['floor'])
    box('논슬립', x + 0.03, YS0 + STW / 2, z + 0.002, 0.04, STW - 0.1, 0.004, M['tact'])
# 계단 옆벽(흰 패널) 및 난간
WALL_Y = YS0 + STW
box('계단사이벽', SX0 + 6.5, WALL_Y + 0.08, CH / 2 + 1.0, 12.0, 0.16, CH + 2.0, M['wall'])
box('계단옆벽', SX0 + 6.5, YS0 - 0.08, CH / 2 + 1.0, 12.0, 0.16, CH + 2.0, M['wall'])
# 에스컬레이터: 경사 몸체 + 유리 난간 + 고무 손잡이
ang = math.atan2(RISE, RUN)
ESY = WALL_Y + 0.16 + ESW / 2 + 0.15
L_es = 7.0
bpy.ops.mesh.primitive_cube_add(size=1, location=(SX0 + 1.2 + L_es / 2 * math.cos(ang), ESY, L_es / 2 * math.sin(ang) + 0.2))
es = bpy.context.active_object; es.name = '에스컬레이터'
es.scale = (L_es, ESW, 0.3); es.rotation_euler[1] = -ang
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
es.data.materials.append(M['esc'])
# 디딤판 홈(경사 위에 얇은 띠 반복)
for k in range(int(L_es / 0.4)):
    d = 0.2 + k * 0.4
    bx = SX0 + 1.2 + d * math.cos(ang); bz = d * math.sin(ang) + 0.36
    box('에스디딤', bx, ESY, bz, 0.36, ESW - 0.1, 0.015, M['steel'])
box('에스컬레이터평부', SX0 + 0.6, ESY, 0.1, 1.2, ESW, 0.2, M['steel'])
for s in (-1, 1):
    yy = ESY + s * (ESW / 2 + 0.06)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(SX0 + 1.2 + L_es / 2 * math.cos(ang), yy, L_es / 2 * math.sin(ang) + 0.75))
    g = bpy.context.active_object; g.name = '에스유리'
    g.scale = (L_es, 0.02, 0.9); g.rotation_euler[1] = -ang
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    g.data.materials.append(M['glass'])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(SX0 + 1.2 + L_es / 2 * math.cos(ang), yy, L_es / 2 * math.sin(ang) + 1.22))
    h = bpy.context.active_object; h.name = '손잡이'
    h.scale = (L_es, 0.08, 0.05); h.rotation_euler[1] = -ang
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    h.data.materials.append(M['rubber'])
box('에스옆벽', SX0 + 6.5, ESY + ESW / 2 + 0.25, CH / 2 + 1.0, 12.0, 0.16, CH + 2.0, M['wall'])
# 계단실 천장(경사)과 조명
bpy.ops.mesh.primitive_cube_add(size=1, location=(SX0 + 0.6 + N * RUN / 2, 0, CH + N * RISE / 2 + 0.2))
rf = bpy.context.active_object; rf.name = '계단실천장'
rf.scale = (N * RUN * 1.25, 2 * SY + 2.0, 0.05); rf.rotation_euler[1] = -ang
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
rf.data.materials.append(M['ceil'])
for k in range(0, N, 3):
    x = SX0 + 1.0 + k * RUN
    z = CH + (k + 1) * RISE + 0.05
    box('계단LED', x, YS0 + STW / 2, z, 0.6, 1.2, 0.01, M['led'])
    area_light(x, YS0 + STW / 2, z - 0.02, 0.6, 1.2, 60)
    box('에스LED', x, ESY, z, 0.6, 0.6, 0.01, M['led'])
    area_light(x, ESY, z - 0.02, 0.6, 0.6, 35)
# 계단 입구 위 천장 개구부 마감(스테인리스 띠)
box('개구부띠', SX0, 0, CH - 0.15, 0.1, 2 * SY, 0.3, M['steel'])

# ---------------- 매달린 안내판 ----------------
def hang_sign(x, y, lines, w=3.0):
    # 판면이 승강장 길이 방향(X)을 바라보게
    box('안내판', x, y, 2.62, 0.12, w, 0.42, M['navy'])
    cyl('매달대', x, y - w / 2 + 0.2, 2.62 + 0.34, 0.015, 0.5, M['steel'], 8)
    cyl('매달대', x, y + w / 2 - 0.2, 2.62 + 0.34, 0.015, 0.5, M['steel'], 8)
    for sgn in (-1, 1):
        text(lines, x + sgn * 0.065, y, 2.62, 0.28, M['sign_txt'], rot=(math.pi / 2, 0, sgn * math.pi / 2))

hang_sign(5.5, 0.0, '나가는 곳  Way Out')
hang_sign(SX0 - 1.0, -0.6, '1·2번 출구  Exit')
hang_sign(32.5, 0.0, '타는 곳  Platform')

# 역명판(스크린도어 머리판 위, 가상 역명)
for sgn in (-1, 1):
    for x in (6.0, 31.0):
        box('역명판', x, sgn * (HW - 0.12), PSD_TOP + 0.62, 2.2, 0.02, 0.36, M['white'])
        text('가람  Garam', x, sgn * (HW - 0.135), PSD_TOP + 0.62, 0.17, M['dark'],
             rot=(math.pi / 2, 0, 0 if sgn > 0 else math.pi))

# 벤치(스테인리스) — 기둥 사이
for x in (2.0, 8.5, 31.5):
    for sgn in (-1, 1):
        for j in range(5):
            box('벤치살', x, sgn * 1.6 + (j - 2) * 0.085, 0.44, 1.8, 0.06, 0.025, M['bench'])
        for dx in (-0.75, 0.75):
            box('벤치다리', x + dx, sgn * 1.6, 0.22, 0.04, 0.04, 0.44, M['bench'])
            box('벤치발', x + dx, sgn * 1.6, 0.01, 0.06, 0.4, 0.02, M['bench'])

# 천장 설비: CCTV·스피커
for i in range(7):
    x = X0 + 3 + i * 7
    if SX0 - 0.5 < x < SX1 + 0.5: continue
    cyl('스피커', x, 0, CH - 0.02, 0.12, 0.04, M['white'], 24)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.08, location=(x + 1.0, 0.8, CH - 0.06))
    d = bpy.context.active_object; d.name = 'CCTV'; d.data.materials.append(M['dark'])

# ---------------- 카메라 ----------------
def cam(name, loc, look, lens=20):
    bpy.ops.object.camera_add(location=loc)
    c = bpy.context.active_object; c.name = name
    d = Vector(look) - Vector(loc)
    c.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    c.data.lens = lens
    return c

C = {
    'c1': cam('c1', (0.0, -0.9, 1.6), (25.0, 0.6, 1.4), 18),
    'c2': cam('c2', (4.3, 0.9, 1.55), (8.6, 5.0, 1.35), 20),
    'c3': cam('c3', (8.0, -0.6, 1.6), (18.0, -0.9, 1.9), 22),
    'c4': cam('c4', (38.5, 0.3, 2.75), (14.0, -0.6, 0.9), 17),
}
sc.camera = C[CAM]

# ---------------- 렌더 설정 ----------------
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.01, 0.01, 0.012, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.3
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
try:
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception:
    pass
sc.cycles.max_bounces = 6
sc.cycles.glossy_bounces = 3
sc.cycles.transmission_bounces = 6
sc.cycles.caustics_reflective = False
sc.cycles.caustics_refractive = False
sc.cycles.sample_clamp_indirect = 4.0
sc.render.resolution_x = 1920; sc.render.resolution_y = 1080
sc.render.resolution_percentage = RES
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - Medium High Contrast'
sc.view_settings.exposure = 0.0
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = os.path.join(OUT, f'r_{CAM}{"_clay" if CLAY else ""}_{SAMPLES}_{RES}.png')
bpy.ops.render.render(write_still=True)
print('WROTE', sc.render.filepath)
