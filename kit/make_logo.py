"""Rebuild the CajaSmart logo (cajasmart.com/CajaSmartLogo.png, 506x213) as a white-on-transparent PNG,
traced from measurements of the original. Icon = band A + band A rotated 180° about the icon center."""
from PIL import Image, ImageDraw, ImageFilter, ImageFont

K = 8                      # draw at 8x native for clean edges
NW, NH = 506, 213
A = [(18, 53), (80, 15.5), (158.5, 60.5), (158.5, 153), (152, 158.5), (101, 130), (56, 155),
     (55, 139), (101, 108.5), (139.5, 128.5), (139.5, 72), (80, 39), (35, 64)]
CX, CY = 80.75, 108.75
B = [(2 * CX - x, 2 * CY - y) for x, y in A]

m = Image.new("L", (NW * K, NH * K), 0)
d = ImageDraw.Draw(m)
for poly in (A, B):
    d.polygon([(x * K, y * K) for x, y in poly], fill=255)
m = m.filter(ImageFilter.GaussianBlur(1.2 * K)).point(lambda v: 255 if v > 128 else 0)  # rounded corners

# wordmark: cap height 45, baseline y=132, spans x=186..504
f = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 62 * K)
txt = "Caja Smart"
w = f.getlength(txt)
size = 62 * K * (318 * K) / w
f = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", int(size))
d = ImageDraw.Draw(m)
d.text((186 * K, 132 * K), txt, font=f, fill=255, anchor="ls")

out = Image.new("RGBA", m.size, (255, 255, 255, 0))
out.putalpha(m)
out = out.resize((NW * 4, NH * 4), Image.LANCZOS)
out.save("./logo.png")
prev = Image.new("RGB", out.size, (37, 40, 54)); prev.paste(out, (0, 0), out)
prev.resize((NW * 2, NH * 2), Image.LANCZOS).save("./logo_preview.png")
print("cap check: text size", round(size / K, 1))
