"""Create an original geometric fixture; no third-party or private artwork."""
import sys
from PIL import Image, ImageDraw

image = Image.new("RGB", (960, 960), "#22182f")
draw = ImageDraw.Draw(image)
for y in range(960):
    draw.line((0, y, 959, y), fill=(34+int(y/70), 24+int(y/110), 47+int(y/80)))
for radius in range(330, 80, -45):
    draw.ellipse((480-radius, 450-radius, 480+radius, 450+radius), outline="#77e3ca", width=7)
draw.polygon([(390, 560), (510, 280), (640, 560)], fill="#efcb80")
draw.rectangle((120, 800, 840, 812), fill="#77e3ca")
draw.text((120, 845), "SYNTHETIC ANALYSIS FIXTURE", fill="#ffffff", font_size=28)
image.save(sys.argv[1])
