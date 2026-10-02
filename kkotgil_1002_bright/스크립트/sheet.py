from PIL import Image,ImageDraw,ImageFont
from fontTools.ttLib import TTFont
KR=ImageFont.truetype('kr.ttf',24); LA=ImageFont.truetype('la.ttf',24)
kcmap=TTFont('kr.ttf').getBestCmap()
def text(dr,xy,s,fill=(20,20,20)):
    x,y=xy
    for ch in s:
        f=KR if ord(ch) in kcmap else LA
        dr.text((x,y),ch,font=f,fill=fill); x+=dr.textlength(ch,font=f)
def grid(cells,cols,cw,ch_,out,title=None):
    rows=(len(cells)+cols-1)//cols; top=50 if title else 0
    c=Image.new('RGB',(cols*cw,rows*(ch_+40)+top),(250,250,248)); dr=ImageDraw.Draw(c)
    if title: text(dr,(12,10),title)
    for k,(path,lab) in enumerate(cells):
        im=Image.open(path).convert('RGB'); im.thumbnail((cw-8,ch_))
        x=(k%cols)*cw+4; y=top+(k//cols)*(ch_+40)
        c.paste(im,(x+(cw-8-im.width)//2,y+40)); text(dr,(x+4,y+6),lab)
    c.save(out,quality=90)
