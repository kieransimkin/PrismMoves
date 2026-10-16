"""Original code-native PrismMoves mark in the established DanceFlow palette."""
from pathlib import Path
from PIL import Image, ImageDraw

target=Path(__file__).resolve().parents[1]/'docs/branding'
target.mkdir(parents=True,exist_ok=True)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128" role="img" aria-labelledby="title desc">
<title id="title">PrismMoves logo</title><desc id="desc">A prism separates one visual input into three evidence paths. Original DanceFlow family mark by Kieran Simkin.</desc>
<rect x="3" y="3" width="122" height="122" rx="30" fill="#171124"/>
<rect x="8" y="8" width="112" height="112" rx="25" fill="none" stroke="#f5ce85" stroke-opacity=".35" stroke-width="1.5"/>
<g fill="none" stroke="#bff7df" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"><path d="M58 35L32 90H84Z M17 63H44"/><path d="M72 58L105 38M76 65H108M80 74L105 94"/></g>
<g fill="#f5ce85"><circle cx="105" cy="38" r="5"/><circle cx="108" cy="65" r="5"/><circle cx="105" cy="94" r="5"/></g></svg>
'''
(target/'logo.svg').write_text(svg,encoding='utf-8')
scale=4
image=Image.new('RGBA',(128*scale,128*scale),(0,0,0,0))
d=ImageDraw.Draw(image)
def box(values): return tuple(int(v*scale) for v in values)
d.rounded_rectangle(box((3,3,125,125)),radius=30*scale,fill='#171124')
d.rounded_rectangle(box((8,8,120,120)),radius=25*scale,outline='#665342',width=2*scale)
for points in [[(58,35),(32,90),(84,90),(58,35)],[(17,63),(44,63)],[(72,58),(105,38)],[(76,65),(108,65)],[(80,74),(105,94)]]:
    scaled=[(x*scale,y*scale) for x,y in points]
    d.line(scaled,fill='#bff7df',width=6*scale,joint='curve')
    for x,y in scaled:
        d.ellipse(box((x-2.75,y-2.75,x+2.75,y+2.75)),fill='#bff7df')
for x,y in [(105,38),(108,65),(105,94)]:
    d.ellipse(box((x-5,y-5,x+5,y+5)),fill='#f5ce85')
image.resize((128,128),Image.Resampling.LANCZOS).save(target/'logo.png')
print(target)
