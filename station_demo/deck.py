"""미팅에서 노트북으로 보여줄 PDF(가로 16:9). 1쪽 클레이|완성, 2~5쪽 컷별, 6쪽 치수 근거."""
from PIL import Image, ImageDraw, ImageFont
import os
D = os.path.dirname(os.path.abspath(__file__))
FB = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 40)
FR = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 28)
FS = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 24)
W, H = 1920, 1080
pages = []

def page(title, sub=None):
    p = Image.new('RGB', (W, H), (18, 18, 20)); d = ImageDraw.Draw(p)
    d.text((60, 30), title, font=FB, fill=(255, 255, 255))
    if sub: d.text((60, 86), sub, font=FR, fill=(190, 190, 195))
    return p, d

def put(p, im, box):
    x0, y0, x1, y1 = box
    r = min((x1 - x0) / im.width, (y1 - y0) / im.height)
    im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    p.paste(im, (x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2))

ld = lambda n: Image.open(f'{D}/{n}').convert('RGB')
p, d = page('지하철 승강장 한 구간 — 구조 먼저, 그다음 재질·조명', '왼쪽: 치수대로 세운 흰 덩어리(구조 확인용)   오른쪽: 같은 모델에 재질·조명을 입힌 렌더')
put(p, ld('r_c1_clay_64_75.png'), (40, 150, 950, 1000)); put(p, ld('r_c1_64_75.png'), (970, 150, 1880, 1000))
d.text((60, 1020), '예시용 가상 역(「가람역」) · 블렌더 · 실제 고객 도면이 아닌 공개 설계기준으로 만든 일반형', font=FS, fill=(170, 170, 175))
pages.append(p)
caps = {
    'c1': ('승강장 길이 방향', '섬식 승강장 폭 10m · 1량 19.5m 기준 문 간격 · 원형 스테인리스 기둥(연단에서 1.5m 이상)'),
    'c2': ('스크린도어 앞', '밀폐형 스크린도어 · 머리판 노선색 띠 · 문 위 칸-문 번호 · 노란 점자선'),
    'c3': ('계단·에스컬레이터', '계단 단높이 16.5cm · 디딤 33cm · 폭 3m · 옆 에스컬레이터 1기'),
    'c4': ('끝에서 본 전체', '천장 LED · 매달린 안내판 · 소화기함 · 벤치 · 선로 쪽 조명'),
}
for c, (t, s) in caps.items():
    p, d = page(t, s); put(p, ld(f'r_{c}_64_75.png'), (40, 140, 1880, 1060)); pages.append(p)
p, d = page('치수 근거와 진행 방식')
lines = [
    '· 승강장 폭·계단: 국토교통부 「도시철도 정거장 및 환승·편의시설 설계 지침」(섬식 최소 8m, 계단 폭 3m, 단 16.5cm·디딤 33cm, 연단 1.5m 안 기둥 금지)',
    '· 승강장 높이 약 1.13~1.15m(레일면 위), 전동차 1량 약 19.5m·한쪽 문 4개 → 문 간격 약 4.9m',
    '· 스크린도어: 서울 지하 역의 밀폐형 기준 · 재질: 밝은 회색 광택 석재 바닥, 흰 천장판, 스테인리스 기둥',
    '',
    '· 실제 작업에서는 받은 도면(DWG) 치수를 그대로 쓰고, 현장 사진으로 재질·안내판을 맞춥니다.',
    '· 렌더용(이미지·영상)과 실시간용(유니티·언리얼·웹)은 같은 모델에서 출발하되, 실시간용은 가볍게 만드는 작업이 더 들어갑니다.',
    '· 도면을 받고 구간을 정하면 그 구간을 3일 안에 무료 샘플로 만들어 드립니다.',
]
for i, l in enumerate(lines):
    d.text((80, 170 + i * 62), l, font=FR, fill=(230, 230, 235))
pages.append(p)
pages[0].save(f'{D}/승강장예시_가람역.pdf', save_all=True, append_images=pages[1:], resolution=150)
print('ok', len(pages))
