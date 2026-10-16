"""Render an actual PixelCue measurement figure from the original synthetic input."""
from pathlib import Path
import json
import sys
from PIL import Image, ImageDraw, ImageFont
from pixelcue.visual_analysis import analyse_image

source, target = map(Path, sys.argv[1:3])
profile = analyse_image(source)
canvas = Image.new('RGB', (1200, 840), '#191426')
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=23)
small = ImageFont.load_default(size=18)
title = ImageFont.load_default(size=34)
draw.text((36, 28), 'PixelCue 0.9.0  >  PrismMoves 0.1.0', font=title, fill='#ffffff')
draw.text((36, 78), 'Synthetic input and actual measured output; semantic model outputs excluded from this figure.', font=small, fill='#d9cce8')
with Image.open(source) as original:
    original.thumbnail((440, 440))
    canvas.paste(original, (36, 133))
draw.text((36, 590), 'Original generated geometric test fixture', font=small, fill='#ffffff')
draw.text((520, 130), 'Measured approximate palette', font=font, fill='#ffffff')
for index, color in enumerate(profile['measurements']['colour']['palette']):
    y = 174+index*53
    draw.rectangle((520, y, 600, y+34), fill=color['hex'])
    draw.text((620, y+4), f"{color['hex']}  {color['sample_fraction']:.1%} of sample", font=small, fill='#ffffff')
measurements = profile['measurements']
rows = [f"Source: {profile['source']['dimensions'][0]} x {profile['source']['dimensions'][1]} pixels",
    f"Analysis: {profile['source']['analysis_dimensions'][0]} x {profile['source']['analysis_dimensions'][1]} pixels",
    f"Grayscale entropy: {measurements['texture']['grayscale_entropy_bits']:.3f} bits",
    f"Mean saturation: {measurements['colour']['mean_saturation']:.3f}",
    f"Edge density: {measurements['composition']['edge_density_above_0_08']:.3f}"]
for index, text in enumerate(rows):
    draw.text((520, 505+index*31), text, font=small, fill='#d9cce8')
for x, label in [(36, 'PixelCue: analyse'), (422, 'PrismMoves: organise'), (808, 'JSON / Markdown / HTML')]:
    draw.rounded_rectangle((x, 708, x+350, 775), radius=12, outline='#77e3ca', width=2)
    draw.text((x+15, 730), label, font=small, fill='#ffffff')
draw.text((36, 800), 'Measurements describe the bounded analysis raster. One image does not establish recurring brand rules.', font=small, fill='#efcb80')
target.parent.mkdir(parents=True, exist_ok=True)
canvas.save(target)
target.with_suffix('.json').write_text(json.dumps(profile, indent=2)+'\n', encoding='utf-8')
print(target)
