"""CajaSmart Reel #2 — '4 cuentas'. 1080x1920 @30fps, exactly 30 s.

Structure follows the iart-ai short-form-video + caption-animation skills:
 - hook text + strongest visual on frame 0 (no fade-up)
 - punch-in pattern interrupt on every cut and on each scene's key beat (~every 2 s)
 - word-by-word captions, active word highlighted, inside the safe band
 - everything frame-driven (deterministic), content kept inside y=250..1450
Usage: python3 render.py [video|audio|sheet|all]
"""
import json, math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

DIR = "./"
W, H, FPS, DUR = 1080, 1920, 30, 30.0
N = int(FPS * DUR)
S = 2  # supersampling factor

# ---- CajaSmart palette (measured from cajasmart.com) ----
BG = (37, 40, 54)        # #252836 page background
CARD = (47, 51, 68)      # white 5% on bg
TRACK = (58, 63, 84)
WAVE = (52, 56, 74)
TXT = (250, 250, 250)    # #FAFAFA
MUTED = (160, 166, 186)
GREEN = (84, 138, 82)    # #548A52 buttons
GREEN_L = (122, 184, 118)
YEL = (255, 228, 146)    # #FFE492 accent
BLUE = (79, 156, 249)    # #4F9CF9 accent
CORAL = (240, 118, 98)   # loss / warning (used sparingly)
DARK = (20, 22, 31)
SKIN = [(232, 184, 148), (198, 139, 89), (241, 201, 165)]

FD = "/usr/share/fonts/opentype/inter/Inter-"
_fc = {}


def font(size, wt="ExtraBold"):
    key = (int(size * S), wt)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(f"{FD}{wt}.otf", key[0])
    return _fc[key]


# ---- easing ----
def cl(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def eoc(x):
    x = cl(x); return 1 - (1 - x) ** 3


def eob(x, k=1.7):
    x = cl(x); return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def mix(c1, c2, a):
    a = cl(a); return tuple(int(c1[i] * a + c2[i] * (1 - a)) for i in range(3))


def punch(t, amp=0.085):
    """spring punch-in: snaps big then settles to 1 (pattern interrupt)."""
    if t < 0: return 1.0
    return 1 + amp * math.exp(-7 * t) * math.cos(11 * t)


# ---- timeline ----
VM = json.load(open(DIR + "voice.json"))
OFF = [0.05] + [0.15] * 6       # voice offset inside each scene (hook starts at once)
PAD = 0.5
starts, t = [], 0.0
for i, m in enumerate(VM):
    starts.append(round(t * FPS) / FPS)
    t = starts[-1] + OFF[i] + m["dur"] + PAD
ends = starts[1:] + [DUR]
assert ends[-1] - starts[-1] > VM[-1]["dur"] + 1.0, "CTA too short"


def wt(sc, word):
    """scene-local start time of the first caption word that starts with `word`."""
    for w in VM[sc]["words"]:
        if w["w"].lower().startswith(word.lower()):
            return OFF[sc] + w["s"]
    raise KeyError(word)


# ---- drawing pen (1x coordinates, drawn at S x) ----
class Pen:
    def __init__(self, img, ox=0, oy=0):
        self.img, self.d, self.ox, self.oy = img, ImageDraw.Draw(img), ox, oy

    def P(self, x, y):
        return ((x + self.ox) * S, (y + self.oy) * S)

    def B(self, x0, y0, x1, y1):
        return [(x0 + self.ox) * S, (y0 + self.oy) * S, (x1 + self.ox) * S, (y1 + self.oy) * S]

    def rr(self, x0, y0, x1, y1, r, fill=None, outline=None, w=0):
        if x1 - x0 < 3 or y1 - y0 < 3: return
        r = max(0, min(r, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1))
        self.d.rounded_rectangle(self.B(x0, y0, x1, y1), radius=r * S, fill=fill, outline=outline, width=int(w * S))

    def el(self, cx, cy, r, fill=None, outline=None, w=0, ry=None):
        ry = r if ry is None else ry
        if r <= 0: return
        self.d.ellipse(self.B(cx - r, cy - ry, cx + r, cy + ry), fill=fill, outline=outline, width=int(w * S))

    def ln(self, pts, fill, w, cap=True):
        self.d.line([self.P(*p) for p in pts], fill=fill, width=int(w * S), joint="curve")
        if cap:
            for p in (pts[0], pts[-1]): self.el(p[0], p[1], w / 2, fill=fill)

    def poly(self, pts, fill):
        self.d.polygon([self.P(*p) for p in pts], fill=fill)

    def arc(self, cx, cy, r, a0, a1, fill, w):
        if a1 <= a0: return
        self.d.arc(self.B(cx - r, cy - r, cx + r, cy + r), a0, a1, fill=fill, width=int(w * S))

    def tx(self, x, y, s, size, fill, wt="ExtraBold", anchor="mm", stroke=0, sfill=DARK):
        self.d.text(self.P(x, y), s, font=font(size, wt), fill=fill, anchor=anchor,
                    stroke_width=int(stroke * S), stroke_fill=sfill)


def tw(s, size, wt="ExtraBold"):
    return font(size, wt).getlength(s) / S


# ---- helpers (shapes) ----
def seg(p, a, b, prog, col, w=8, arrow=False, dash=0):
    """line from a to b drawn up to `prog` (0..1); optional arrowhead/dashes."""
    prog = cl(prog)
    if prog <= 0: return
    x = a[0] + (b[0] - a[0]) * prog; y = a[1] + (b[1] - a[1]) * prog
    if dash:
        L = math.hypot(x - a[0], y - a[1]); n = int(L // (dash * 2)) + 1
        ux, uy = ((b[0] - a[0]), (b[1] - a[1])); LL = math.hypot(ux, uy) or 1; ux, uy = ux / LL, uy / LL
        for i in range(n):
            s0 = i * dash * 2; s1 = min(L, s0 + dash)
            if s1 > s0: p.ln([(a[0] + ux * s0, a[1] + uy * s0), (a[0] + ux * s1, a[1] + uy * s1)], col, w)
    else:
        p.ln([a, (x, y)], col, w)
    if arrow and prog > 0.85:
        ang = math.atan2(b[1] - a[1], b[0] - a[0]); L = 26
        p.poly([(x + math.cos(ang) * 8, y + math.sin(ang) * 8),
                (x - L * math.cos(ang - 0.5), y - L * math.sin(ang - 0.5)),
                (x - L * math.cos(ang + 0.5), y - L * math.sin(ang + 0.5))], col)


def node(p, cx, cy, r, col, label="", tcol=TXT, size=None):
    p.el(cx, cy, r, fill=col)
    if label: p.tx(cx, cy, label, size or r * 0.9, tcol, "Black")


def check(p, cx, cy, r, prog, col=GREEN, mark=TXT):
    if prog <= 0: return
    p.el(cx, cy, r * eob(prog), fill=col)
    a, b, c = (cx - r * .45, cy + r * .02), (cx - r * .12, cy + r * .35), (cx + r * .5, cy - r * .35)
    q = cl((prog - 0.3) / 0.7)
    if q > 0:
        if q < 0.4: p.ln([a, (a[0] + (b[0] - a[0]) * q / .4, a[1] + (b[1] - a[1]) * q / .4)], mark, r * .2)
        else:
            k = (q - .4) / .6
            p.ln([a, b, (b[0] + (c[0] - b[0]) * k, b[1] + (c[1] - b[1]) * k)], mark, r * .2)


def cross(p, cx, cy, r, prog, col=CORAL):
    if prog <= 0: return
    k = eob(prog); rr = r * k
    p.ln([(cx - rr, cy - rr), (cx + rr, cy + rr)], col, r * .32)
    p.ln([(cx - rr, cy + rr), (cx + rr, cy - rr)], col, r * .32)


def lock(p, cx, cy, s, col=YEL):
    p.arc(cx, cy - s * .35, s * .38, 180, 360, col, s * .14)
    p.rr(cx - s * .55, cy - s * .35, cx + s * .55, cy + s * .5, s * .12, fill=col)
    p.el(cx, cy + s * .05, s * .1, fill=BG)


def person(p, cx, base, s=1.0, skin=SKIN[0], shirt=(64, 70, 94), apron=GREEN, arm=0.0):
    hr = 40 * s
    p.rr(cx - 62 * s, base - 165 * s, cx + 62 * s, base, 40 * s, fill=shirt)          # body
    p.rr(cx - 40 * s, base - 128 * s, cx + 40 * s, base - 8 * s, 12 * s, fill=apron)   # apron
    p.ln([(cx - 30 * s, base - 128 * s), (cx - 18 * s, base - 160 * s)], apron, 7 * s)
    p.ln([(cx + 30 * s, base - 128 * s), (cx + 18 * s, base - 160 * s)], apron, 7 * s)
    sx, sy = cx + 55 * s, base - 140 * s  # right arm (raise with arm>0)
    p.ln([(sx, sy), (sx + 20 * s, sy + 60 * s - arm * 120 * s)], shirt, 22 * s)
    p.ln([(cx - 55 * s, base - 140 * s), (cx - 72 * s, base - 80 * s)], shirt, 22 * s)
    p.el(cx, base - 165 * s - hr + 4 * s, hr, fill=skin)                                # head
    p.el(cx - 13 * s, base - 165 * s - hr + 2 * s, 4.5 * s, fill=DARK)
    p.el(cx + 13 * s, base - 165 * s - hr + 2 * s, 4.5 * s, fill=DARK)


def robot(p, cx, base, s=1.0, wave=0.0, blink=False):
    body, head = (76, 84, 110), (96, 106, 138)
    p.rr(cx - 60 * s, base - 150 * s, cx + 60 * s, base, 22 * s, fill=body)
    p.el(cx, base - 80 * s, 18 * s, fill=GREEN_L)
    p.rr(cx - 70 * s, base - 260 * s, cx + 70 * s, base - 160 * s, 28 * s, fill=head)
    p.ln([(cx, base - 260 * s), (cx, base - 292 * s)], head, 6 * s)
    p.el(cx, base - 298 * s, 10 * s, fill=YEL)
    ey = base - 212 * s
    for dx in (-28, 28):
        if blink: p.ln([(cx + (dx - 12) * s, ey), (cx + (dx + 12) * s, ey)], BLUE, 6 * s)
        else: p.el(cx + dx * s, ey, 12 * s, fill=BLUE)
    p.ln([(cx - 22 * s, base - 182 * s), (cx + 22 * s, base - 182 * s)], DARK, 5 * s)
    p.ln([(cx - 60 * s, base - 120 * s), (cx - 90 * s, base - 60 * s)], body, 20 * s)
    a = -1.2 - 0.5 * math.sin(wave)
    x0, y0 = cx + 60 * s, base - 125 * s
    p.ln([(x0, y0), (x0 + math.cos(a) * 85 * s, y0 + math.sin(a) * 85 * s)], body, 20 * s)


import os
LOGO = None
for _lp in (DIR + "logo.png",):
    if os.path.exists(_lp):
        LOGO = Image.open(_lp).convert("RGBA")
        _bb = LOGO.getbbox()
        LOGO = LOGO.crop(_bb) if _bb else LOGO


def logo(p, x, y, h, anchor="lm"):
    """CajaSmart logo (from cajasmart.com) at height h; anchor lm = left-middle, mm = centered."""
    if LOGO is None:
        icon_hex(p, x + 20, y, 20 * h / 44, GREEN_L, 5)
        p.tx(x + 50 * h / 44, y, "Caja Smart", 30 * h / 44, TXT, "Bold", anchor="lm")
        return
    w = LOGO.width * h / LOGO.height
    im = LOGO.resize((int(w * S), int(h * S)), Image.LANCZOS)
    x0 = x - (w / 2 if anchor == "mm" else 0)
    p.img.paste(im, (int((x0 + p.ox) * S), int((y - h / 2 + p.oy) * S)), im)


def icon_hex(p, cx, cy, r, col, w=5):
    pts = [(cx + r * math.cos(math.radians(60 * i - 90)), cy + r * math.sin(math.radians(60 * i - 90))) for i in range(7)]
    p.ln(pts, col, w)


def icon_cup(p, cx, cy, s=1.0, fill=1.0, steam_t=0.0):
    top, bot = cy - 95 * s, cy + 105 * s
    p.poly([(cx - 78 * s, top), (cx + 78 * s, top), (cx + 58 * s, bot), (cx - 58 * s, bot)], TXT)
    p.rr(cx - 90 * s, top - 26 * s, cx + 90 * s, top + 4 * s, 10 * s, fill=(225, 225, 232))
    sl0, sl1 = cy - 30 * s, cy + 40 * s
    p.poly([(cx - 72 * s, sl0), (cx + 72 * s, sl0), (cx + 65 * s, sl1), (cx - 65 * s, sl1)], GREEN)
    icon_hex(p, cx, (sl0 + sl1) / 2, 20 * s, TXT, 5 * s)
    for i, dx in enumerate((-30, 0, 30)):
        ph = steam_t * 3 + i
        pts = [(cx + dx * s + 10 * s * math.sin(ph + k * 0.9), top - 40 * s - k * 14 * s) for k in range(6)]
        p.ln(pts, mix(TXT, BG, 0.35 + 0.25 * math.sin(ph)), 5 * s)


def icon_tag(p, cx, cy, w, h, col, label, size, tcol=TXT):
    x0, x1 = cx - w / 2, cx + w / 2
    p.poly([(x0 + h * .35, cy - h / 2), (x1, cy - h / 2), (x1, cy + h / 2), (x0 + h * .35, cy + h / 2), (x0, cy)], col)
    p.el(x0 + h * .38, cy, h * .09, fill=BG)
    p.tx(cx + h * .17, cy + 2, label, size, tcol, "Black")


def icon_coin(p, cx, cy, r, col=YEL):
    p.el(cx, cy, r, fill=col)
    p.el(cx, cy, r * .72, outline=mix(DARK, col, .25), w=r * .12)
    p.tx(cx, cy + 1, "$", r * 1.0, mix(DARK, col, .6), "Black")


def bubble(p, x0, y0, x1, y1, tail, col, outline=None):
    p.poly([(tail[0] - 26, y1 - 2), (tail[0] + 26, y1 - 2), tail], outline or col)
    p.rr(x0, y0, x1, y1, 34, fill=col, outline=outline, w=6 if outline else 0)
    if outline:
        p.poly([(tail[0] - 20, y1 - 8), (tail[0] + 20, y1 - 8), (tail[0], tail[1] - 12)], col)


def pill(p, cx, cy, txt, size, bg, fg=TXT, padx=26, h=None, wt="Black"):
    w = tw(txt, size, wt) + padx * 2; h = h or size * 1.6
    p.rr(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, h / 2, fill=bg)
    p.tx(cx, cy + 1, txt, size, fg, wt)
    return w


def bracket(p, x0, x1, y, col, prog, down=True, label="", size=30):
    if prog <= 0: return
    d = 18 if down else -18
    xm = (x0 + x1) / 2; q = eoc(prog)
    p.ln([(xm - (xm - x0) * q, y), (xm + (x1 - xm) * q, y)], col, 6)
    if q > .95:
        p.ln([(x0, y - d), (x0, y)], col, 6); p.ln([(x1, y - d), (x1, y)], col, 6)
    if label and prog > .5:
        p.tx(xm, y + (34 if down else -34), label, size, mix(col, BG, eoc((prog - .5) * 2)), "Bold")


def money(v):
    return f"${v:,.2f}"


# ===================== STYLE 3: "how it works inside" explainer =====================
# Reference: @resultaaquee — near-black grid world, one continuous camera, macro cross-section,
# glowing gold arcs, tiny spaced-caps callouts with leader lines, a stat card, top captions with
# the current word in white and key words inside a blue "selection box", share/follow sheet ending.

BG = (11, 13, 20)
GRID = (21, 24, 35)
PART = (70, 78, 104)
CARD = (25, 28, 40)
CARD2 = (34, 38, 53)
GRAYC = (128, 134, 150)
GOLD = YEL
WALL = (233, 234, 239)
COFFEE = (92, 58, 34)
MILK = (214, 190, 156)
FOAM = (242, 232, 214)
LID = (196, 200, 212)

SX, SY = 540, 860                       # screen point the camera looks at
CAM = [(1.0, 0, -80), (1.15, 80, -40), (1.2, 100, -20), (0.72, 0, 40), (0.62, 0, 250), (0.62, 0, 250), (0.45, 0, 520)]


def smooth(x):
    x = cl(x); return x * x * (3 - 2 * x)


def scene_at(t):
    sc = max(i for i in range(7) if starts[i] <= t + 1e-6)
    return sc, t - starts[sc]


def camera(t):
    sc, ls = scene_at(t)
    a = CAM[max(0, sc - 1)]; b = CAM[sc]
    k = smooth(ls / 0.75) if sc > 0 else 1.0
    s = a[0] + (b[0] - a[0]) * k
    cx = a[1] + (b[1] - a[1]) * k
    cy = a[2] + (b[2] - a[2]) * k
    return s * (1 + 0.022 * ls), cx, cy        # slow continuous push-in


class Cam:
    def __init__(self, s, cx, cy):
        self.s, self.cx, self.cy = s, cx, cy

    def X(self, x): return SX + (x - self.cx) * self.s

    def Y(self, y): return SY + (y - self.cy) * self.s

    def P(self, x, y): return (self.X(x), self.Y(y))


def glow(p, pts, col, w):
    p.ln(pts, mix(col, BG, 0.12), w * 4.5)
    p.ln(pts, mix(col, BG, 0.30), w * 2.4)
    p.ln(pts, col, w)


def bez(a, c, b, n=24, upto=1.0):
    out = []
    for i in range(n + 1):
        u = i / n * upto
        out.append(((1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u * u * b[0],
                    (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u * u * b[1]))
    return out


def draw_bg(p, t):
    for x in range(0, W + 1, 90): p.ln([(x, 0), (x, H)], GRID, 2, cap=False)
    for y in range(0, H + 1, 90): p.ln([(0, y), (W, y)], GRID, 2, cap=False)
    for i in range(70):                      # drifting dust particles
        x = (i * 137.5 + 20 * math.sin(t * 0.3 + i)) % W
        y = (i * 271.3 - t * (10 + 8 * (i % 3))) % H
        r = 1.6 + (i % 4) * 0.7
        p.el(x, y, r, fill=mix(PART, BG, 0.45 + 0.4 * math.sin(t * 1.3 + i * 0.7)))


# ---- the coffee cup in cross-section (world coords: y -300 top .. 300 bottom) ----
def hw_out(y): return 210 + (160 - 210) * (y + 300) / 600


def hw_in(y): return hw_out(y) - 16


def band(c, y0, y1, col):
    return [c.P(-hw_in(y0), y0), c.P(hw_in(y0), y0), c.P(hw_in(y1), y1), c.P(-hw_in(y1), y1)], col


def cup(p, c, t, dim=None, hi=None, steam=1.0):
    dim = dim or {}
    d = lambda part, col: mix(col, BG, dim[part]) if part in dim else col
    s = c.s
    # outer wall + sleeve
    p.poly([c.P(-hw_out(-300), -300), c.P(hw_out(-300), -300), c.P(hw_out(300), 300), c.P(-hw_out(300), 300)], d("cup", WALL))
    for sg in (-1, 1):
        p.poly([c.P(sg * hw_out(-70), -70), c.P(sg * (hw_out(-70) + 6), -70), c.P(sg * (hw_out(70) + 6), 70), c.P(sg * hw_out(70), 70)],
               d("cup", GREEN))
    # inner layers
    for part, y0, y1, col in (("foam", -232, -150, FOAM), ("milk", -150, 40, MILK), ("coffee", 40, 284, COFFEE)):
        pts, colr = band(c, y0, y1, col)
        p.poly(pts, d(part, colr))
    p.poly([c.P(-hw_in(-300), -300), c.P(hw_in(-300), -300), c.P(hw_in(-232), -232), c.P(-hw_in(-232), -232)], BG)  # air gap
    # bubbles in milk/foam (life)
    for i in range(7):
        bx = -120 + i * 40 + 6 * math.sin(t * 2 + i); by = -195 + 10 * math.sin(t * 1.7 + i * 2)
        p.el(c.X(bx), c.Y(by), 5 * s, fill=d("foam", (250, 244, 232)))
    # lid
    p.rr(c.X(-228), c.Y(-334), c.X(228), c.Y(-298), 14 * s, fill=d("cup", LID))
    p.rr(c.X(-70), c.Y(-352), c.X(70), c.Y(-330), 10 * s, fill=d("cup", LID))
    # highlight outline for the part being priced
    if hi:
        y0, y1 = {"coffee": (40, 284), "milk": (-150, 40), "cup": (-352, 300)}[hi]
        pts = [c.P(-hw_out(y0) - 6, y0 - 6), c.P(hw_out(y0) + 6, y0 - 6), c.P(hw_out(y1) + 6, y1 + 6), c.P(-hw_out(y1) - 6, y1 + 6)]
        glow(p, pts + [pts[0]], GOLD, 4 * max(0.6, s))
    if steam > 0.01:
        for i, dx in enumerate((-60, 0, 60)):
            ph = t * 2.2 + i * 1.3
            pts = [(c.X(dx + 16 * math.sin(ph + k * 0.8)), c.Y(-372 - k * 24)) for k in range(5)]
            p.ln(pts, mix(TXT, BG, (0.25 + 0.15 * math.sin(ph)) * steam), 6 * s)


def callout(p, ax, ay, lx, ly, label, value, k, col=GOLD):
    """tiny spaced-caps label + value with a thin leader line from anchor (ax, ay)."""
    if k <= 0: return
    q = eoc(k)
    p.el(ax, ay, 7, fill=col)
    ex = ax + (lx - 14 - ax) * q
    p.ln([(ax, ay), (ex, ly)], mix(TXT, BG, 0.75), 2.5)
    if q > 0.6:
        a = (q - 0.6) / 0.4
        p.tx(lx, ly - 22, "\u2009".join(label), 26, mix(TXT, BG, a), "SemiBold", anchor="lm")
        p.tx(lx, ly + 24, value, 50, mix(col, BG, a), "Black", anchor="lm")


def icon_node(p, x, y, r, kind, k, col=GOLD):
    if k <= 0: return
    r *= eob(k)
    p.el(x, y, r, fill=CARD2, outline=col, w=4)
    s = r / 60
    if kind == "luz":
        p.el(x, y - 8 * s, 20 * s, fill=YEL); p.rr(x - 9 * s, y + 10 * s, x + 9 * s, y + 24 * s, 3 * s, fill=LID)
    elif kind == "agua":
        p.poly([(x, y - 28 * s), (x + 18 * s, y + 4 * s), (x - 18 * s, y + 4 * s)], BLUE); p.el(x, y + 6 * s, 18 * s, fill=BLUE)
    elif kind == "renta":
        p.poly([(x - 26 * s, y - 2 * s), (x, y - 26 * s), (x + 26 * s, y - 2 * s)], TXT)
        p.rr(x - 19 * s, y - 2 * s, x + 19 * s, y + 24 * s, 3 * s, fill=TXT); p.rr(x - 6 * s, y + 8 * s, x + 6 * s, y + 24 * s, 2 * s, fill=CARD2)
    elif kind == "equipo":
        for dx, sh in ((-14, 0.85), (14, 1.0)):
            p.el(x + dx * s, y - 12 * s * sh, 10 * s * sh, fill=TXT)
            p.rr(x + dx * s - 14 * s * sh, y + 2 * s, x + dx * s + 14 * s * sh, y + 26 * s, 10 * s, fill=GREEN_L if dx > 0 else TXT)


def stat_card(p, y, big, l1, l2, k, col=GOLD):
    if k <= 0: return
    dy = 30 * (1 - eoc(k))
    p.rr(110, y + dy, 950, y + 170 + dy, 26, fill=CARD, outline=CARD2, w=3)
    p.tx(150, y + 85 + dy, big, 96, mix(col, CARD, eoc(k)), "Black", anchor="lm")
    bx = 150 + tw(big, 96, "Black") + 34
    p.tx(bx, y + 66 + dy, l1, 38, mix(TXT, CARD, eoc(k)), "Bold", anchor="lm")
    p.tx(bx, y + 112 + dy, l2, 28, mix(GRAYC, CARD, eoc(k)), "Medium", anchor="lm")


def logo_icon(p, cx, cy, h):
    if LOGO is None: icon_hex(p, cx, cy, h / 2, TXT, 5); return
    ic = LOGO.crop((0, 0, int(LOGO.width * 161 / 506), LOGO.height))
    w = ic.width * h / ic.height
    im = ic.resize((int(w * S), int(h * S)), Image.LANCZOS)
    p.img.paste(im, (int((cx - w / 2) * S), int((cy - h / 2) * S)), im)


# ---- per-scene world + overlay ----
def world(p, t):
    sc, ls = scene_at(t)
    c = Cam(*camera(t))
    dim, hi, steam = {}, None, 1.0
    if sc == 1:
        to = wt(1, "olvidan") - 0.05
        f = 0.3 if ls < to else 0.3 + 0.7 * eoc((ls - to) / 0.3)
        dim = {"cup": f, "milk": f, "foam": f}
        steam = 0.0
    if sc == 2:
        steam = 0.0
        for word, part in (("$0.25", "cup"), ("Leche", "milk"), ("café", "coffee")):
            if ls >= wt(2, word) - 0.05: hi = part; break
    if sc == 3: steam = smooth((ls - 0.3) / 0.5)
    cup(p, c, t, dim, hi, steam)
    if sc == 3:                               # invisible costs flowing into the cup
        nodes = (("luz", -340, -230, "luz"), ("agua", -390, 20, "luz"), ("renta", -340, 255, "luz"), ("equipo", 370, 40, "equipo"))
        for i, (kind, x, y, grp) in enumerate(nodes):
            t0 = (wt(3, "Luz") + i * 0.22) if grp == "luz" else wt(3, "equipo") - 0.2
            k = (ls - t0) / 0.35
            a = c.P(x, y); b = c.P(-150 if x < 0 else 150, y * 0.3)
            mid = ((a[0] + b[0]) / 2, min(a[1], b[1]) - 90 * c.s)
            if k > 0.3: glow(p, bez(a, mid, b, upto=cl((k - 0.3) / 0.7)), GOLD, 3.5)
            if k > 1.0:                        # energy dot travelling along the arc
                u = ((ls - t0) * 0.9) % 1.0
                q = bez(a, mid, b, n=1, upto=u)[-1]
                p.el(q[0], q[1], 8, fill=TXT)
            icon_node(p, a[0], a[1], 60 * c.s / 0.72, kind, k)


def overlay(p, t):
    sc, ls = scene_at(t)
    c = Cam(*camera(t))
    if sc == 1:
        to = wt(1, "olvidan") - 0.05
        callout(p, c.X(60), c.Y(160), 735, c.Y(160), "CAFÉ", "✓", (ls - wt(1, "café")) / 0.35, GREEN_L)
        if ls > to:
            for j, (x, y) in enumerate(((0, -190), (60, -50), (-hw_out(-200) + 8, -200))):
                k = eob((ls - to - j * 0.1) / 0.3)
                if k > 0: node(p, c.X(x), c.Y(y), 30 * k, CORAL, "?", DARK, 40 * k)
    if sc == 2:
        for word, (wx, wy), ly, lab, val in (("café", (60, 160), None, "CAFÉ", "$0.40"),
                                             ("Leche", (40, -55), None, "LECHE", "$0.30"),
                                             ("Vaso", (200, -316), None, "VASO Y TAPA", "$0.25")):
            callout(p, c.X(wx), c.Y(wy), 715, c.Y(wy), lab, val, (ls - wt(2, word) + 0.05) / 0.4)
    if sc == 3:
        k1 = (ls - wt(3, "renta") + 0.1) / 0.4; k2 = (ls - wt(3, "equipo") + 0.0) / 0.4
        for k, x, lab, val in ((k1, 250, "LUZ · AGUA · RENTA", "$0.50"), (k2, 815, "EQUIPO", "$0.60")):
            if k > 0:
                a = eoc(k)
                p.tx(x, 1170, lab, 28, mix(TXT, BG, a), "SemiBold")
                p.tx(x, 1225, val, 54, mix(GOLD, BG, a), "Black")
    if sc == 4:
        vals = [0.40, 0.30, 0.25, 0.50, 0.60]
        t0 = wt(4, "$2.05") - 0.6
        total = 0.0
        for i, v in enumerate(vals):          # reading chips light up one by one (sensor-grid feel)
            k = (ls - 0.1 - i * 0.12) / 0.25
            on = ls > t0 + i * 0.12
            x = 175 + i * 170
            if k > 0:
                p.rr(x - 72, 955, x + 72, 1035, 16, fill=CARD2 if not on else mix(GOLD, BG, 0.22), outline=GOLD if on else CARD2, w=3)
                p.tx(x, 996, f"${v:.2f}", 36, TXT if on else GRAYC, "Bold")
            if on: total += v * eoc((ls - t0 - i * 0.12) / 0.25)
        stat_card(p, 1090, f"${total:.2f}", "costo real", "por taza · ejemplo", (ls - 0.2) / 0.4)
    if sc == 5:
        tp, tq = wt(5, "$2.50") - 0.1, wt(5, "$0.45") - 0.15
        k = eoc((ls - 0.05) / 0.35)
        p.rr(110, 960, 110 + 840 * k, 1050, 18, fill=CARD2)
        g = eoc((ls - tp) / 0.5)
        if g > 0:
            wc = 840 * (2.05 / 2.50)
            p.rr(110, 960, 110 + wc * g, 1050, 18, fill=TRACK)
            if g > 0.8: p.tx(110 + wc / 2, 1005, "costo $2.05", 34, TXT, "Bold")
            p.rr(118 + wc, 960, 118 + wc + (840 - wc - 8) * eoc((ls - tq) / 0.35), 1050, 18, fill=GREEN_L)
            p.tx(950, 920, "precio $2.50", 34, mix(TXT, BG, g), "Bold", anchor="rm")
        stat_card(p, 1110, "$0.45", "te quedan", "por taza", (ls - tq) / 0.4, GREEN_L)
    if sc == 6:
        lg = eob((ls - 0.1) / 0.4)
        if lg > 0.05: logo(p, 540, 850, 100 * lg, anchor="mm")
        k = eoc((ls - 0.35) / 0.45)
        top = 1440 - 410 * k
        if k > 0:
            p.rr(100, top, 960, top + 410, 40, fill=CARD, outline=CARD2, w=3)
            p.rr(480, top + 18, 580, top + 26, 4, fill=GRAYC)
            p.el(200, top + 120, 66, fill=CARD2, outline=GOLD, w=4); logo_icon(p, 200, top + 120, 70)
            p.tx(290, top + 98, "Caja Smart", 44, TXT, "Black", anchor="lm")
            p.tx(290, top + 146, "Tu POS de confianza", 30, GRAYC, "Medium", anchor="lm")
            tf = wt(6, "¡Sí") + 0.15
            followed = ls >= tf
            by = top + 300
            p.rr(150, by - 52, 910, by + 52, 22, fill=CARD2 if followed else BLUE)
            p.tx(530, by, "✓ Siguiendo" if followed else "Seguir", 42, TXT, "Black")
            if tf - 0.35 < ls < tf + 0.25:     # finger tap ripple (gone once the tap is done)
                r = cl((ls - tf + 0.35) / 0.6)
                p.el(700, by + 12, 18 + 70 * r, outline=mix(TXT, BG, 1 - r), w=5)
                if r < 1: p.el(700, by + 12, 18, fill=mix(TXT, BG, 0.8))


# ---- captions: top, 1–3 words, current word white, key words in a selection box ----
KEYS = {"café", "solo", "demás", "dentro", "$0.40", "$0.30", "$0.25", "$0.50", "$0.60", "$2.05", "$2.50", "$0.45",
        "cajasmart", "¡síguenos", "real", "equipo"}
PAGES = []
for _sc, _m in enumerate(VM):
    pages, cur = [], []
    for w in _m["words"]:
        cur.append(w)
        if len(cur) == 3 or w["w"][-1] in ".,:!?":
            pages.append(cur); cur = []
    if cur: pages.append(cur)
    PAGES.append(pages)


def draw_caption(p, sc, ls):
    vt = ls - OFF[sc]
    pages = PAGES[sc]; cur = None
    for i, pg in enumerate(pages):
        nxt = pages[i + 1][0]["s"] if i + 1 < len(pages) else 99
        if pg[0]["s"] - 0.05 <= vt < nxt - 0.05: cur = pg
    if cur is None:
        if vt > pages[-1][-1]["e"]: cur = pages[-1]
        else: return
    size, y = 64, 362
    words = [w["w"].strip(",:") for w in cur]
    widths = [tw(w, size, "Bold") for w in words]
    space = tw(" ", size, "Bold")
    x = 530 - (sum(widths) + space * (len(words) - 1)) / 2
    started = [w for w in cur if vt >= w["s"] - 0.05]
    last = started[-1] if started else None
    for w, ww, info in zip(words, widths, cur):
        if vt >= info["s"] - 0.05:
            key = w.lower().strip(".?!") in KEYS
            isnow = info is last
            col = TXT if (isnow or key) else GRAYC
            if key:
                k = eob((vt - info["s"] + 0.05) / 0.2)
                pad = 12
                x0, x1, y0, y1 = x - pad, x + ww + pad, y - 44, y + 40
                p.rr(x0, y0, x0 + (x1 - x0) * k, y1, 4, outline=BLUE, w=3)
                if k > 0.9:
                    for hx, hy in ((x0, y0), (x1, y1)):
                        p.el(hx, hy, 7, fill=TXT, outline=BLUE, w=3)
            p.tx(x + ww / 2, y, w, size, col, "Bold")
        x += ww + space


def draw_chrome(p):
    logo(p, 92, 282, 52)


HOOK0 = None


def render_frame(f):
    t = f / FPS
    sc, ls = scene_at(t)
    img = Image.new("RGB", (W * S, H * S), BG)
    p = Pen(img)
    draw_bg(p, t)
    world(p, t)
    overlay(p, t)
    draw_chrome(p)
    draw_caption(p, sc, ls)
    return img.reduce(S)


def video():
    global HOOK0
    HOOK0 = render_frame(0)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "17", "-pix_fmt", "yuv420p", DIR + "video_only.mp4"], stdin=subprocess.PIPE)
    for f in range(N):
        fr = render_frame(f)
        tail = N - 1 - f
        if tail < 8: fr = Image.blend(fr, HOOK0, (8 - tail) / 8 * 0.85)
        proc.stdin.write(fr.tobytes())
        if f % 150 == 0: print("frame", f, flush=True)
    proc.stdin.close(); proc.wait()


# ---------------- audio ----------------
SR = 48000


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def music():
    n = int(DUR * SR); out = np.zeros(n)
    bpm = 100; beat = 60 / bpm
    rng = np.random.default_rng(7)
    # kick / snare / hats
    for b in range(int(DUR / beat) + 1):
        i = int(b * beat * SR)
        L = int(0.35 * SR); tt = np.arange(L) / SR
        kick = np.sin(2 * np.pi * (45 * tt + (120 - 45) * 0.04 * (1 - np.exp(-tt / 0.04)))) * env_exp(L, 0.12)
        out[i:i + L] += 0.9 * kick[:max(0, min(L, n - i))]
        if b % 2 == 1:
            Ls = int(0.2 * SR)
            sn = (rng.standard_normal(Ls) * 0.6 + np.sin(2 * np.pi * 190 * np.arange(Ls) / SR)) * env_exp(Ls, 0.05)
            out[i:i + Ls] += 0.35 * sn[:max(0, min(Ls, n - i))]
        for h in (0, 0.5):
            j = int((b + h) * beat * SR); Lh = int(0.05 * SR)
            hh = np.diff(rng.standard_normal(Lh + 1)) * env_exp(Lh, 0.012)
            out[j:j + Lh] += 0.12 * hh[:max(0, min(Lh, n - j))]
    # chords Am F C G (one bar each) + bass
    prog = [[57, 60, 64], [53, 57, 60], [48, 55, 60], [55, 59, 62]]
    bar = 4 * beat; tt = np.arange(n) / SR
    pad = np.zeros(n); bass = np.zeros(n)
    for k in range(int(DUR / bar) + 1):
        a, b = int(k * bar * SR), min(n, int((k + 1) * bar * SR))
        if a >= n: break
        seg_t = tt[a:b] - tt[a]
        envp = np.minimum(1, seg_t / 0.25) * np.minimum(1, (bar - seg_t) / 0.2)
        for note in prog[k % 4]:
            fq = 440 * 2 ** ((note - 69) / 12)
            for det in (-0.004, 0.004):
                ph = 2 * np.pi * fq * (1 + det) * tt[a:b]
                pad[a:b] += (2 / np.pi) * np.arcsin(np.sin(ph)) * envp * 0.09
        fr = 440 * 2 ** ((prog[k % 4][0] - 12 - 69) / 12)
        for half in (0, 2):
            s0 = a + int(half * beat * SR); L = int(1.6 * beat * SR)
            s1 = min(n, s0 + L)
            if s0 >= n: continue
            tb = np.arange(s1 - s0) / SR
            bass[s0:s1] += (np.sin(2 * np.pi * fr * tb) + 0.25 * np.sign(np.sin(2 * np.pi * fr * tb))) * env_exp(s1 - s0, 0.5) * 0.35
    pad = sosfilt(butter(2, 2200, fs=SR, output="sos"), pad)
    bass = sosfilt(butter(2, 400, fs=SR, output="sos"), bass)
    return out * 0.55 + pad + bass


def whooshes():
    n = int(DUR * SR); out = np.zeros(n)
    rng = np.random.default_rng(3)
    for s in starts[1:]:
        c = int(s * SR); a, b = c - int(0.28 * SR), c + int(0.22 * SR)
        L = b - a; noise = rng.standard_normal(L)
        y = np.zeros(L); prev = 0.0
        for i in range(L):  # one-pole lowpass with sweeping cutoff
            x = i / L; fc = 300 + 5500 * math.sin(math.pi * x) ** 2
            al = 1 - math.exp(-2 * math.pi * fc / SR)
            prev += al * (noise[i] - prev); y[i] = prev
        e = np.sin(np.pi * np.arange(L) / L) ** 2
        out[max(0, a):b] += (y * e * 0.55)[max(0, -a):]
    return out


def audio():
    n = int(DUR * SR)
    voice = np.zeros(n)
    for i, m in enumerate(VM):
        v, sr = sf.read(DIR + f"v{i}.wav")
        v = resample_poly(v, SR // 1000, sr // 1000) if sr != SR else v
        a = int((starts[i] + OFF[i]) * SR)
        voice[a:a + len(v)] += v[:max(0, min(len(v), n - a))]
    voice /= np.max(np.abs(voice)) + 1e-9
    mus = music(); mus /= np.max(np.abs(mus)) + 1e-9
    # duck music under voice (fast attack, slow release)
    hop = 480; e = np.array([np.max(np.abs(voice[i:i + hop])) for i in range(0, n, hop)])
    sm = np.zeros_like(e); g = 0.0
    for i, x in enumerate(e):
        g = g + (x - g) * (0.6 if x > g else 0.04); sm[i] = g
    duck = 1 - 0.72 * np.clip(sm / 0.35, 0, 1)
    duck = np.repeat(duck, hop)[:n]
    mus *= 0.42 * duck
    fade = np.ones(n); fl = int(1.8 * SR); fade[-fl:] = np.linspace(1, 0, fl) ** 1.5
    mixd = voice * 0.85 + mus * fade + whooshes() * 0.35 * fade
    mixd /= np.max(np.abs(mixd)) / 0.9
    sf.write(DIR + "mix.wav", np.stack([mixd, mixd], 1).astype(np.float32), SR, subtype="FLOAT")


def mux():
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", DIR + "video_only.mp4", "-i", DIR + "mix.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-t", "30", "-movflags", "+faststart",
                    DIR + "cajasmart_costo_cafe_ES.mp4"], check=True)


def sheet():
    """contact sheet: one frame per scene (+ hook frame 0), with the 250/1450 safe lines."""
    pick = [0] + [int((starts[i] + min(ends[i] - starts[i] - 0.15, OFF[i] + VM[i]["dur"])) * FPS) for i in range(7)]
    pick[-1] = int((starts[6] + 5.0) * FPS)
    tw_, th_ = 360, 640
    sh = Image.new("RGB", (tw_ * 4, th_ * 2), (0, 0, 0))
    for k, f in enumerate(pick):
        fr = render_frame(f)
        d = ImageDraw.Draw(fr)
        for y in (250, 1450): d.line([(0, y), (W, y)], fill=(255, 0, 80), width=4)
        d.line([(960, 250), (960, 1450)], fill=(255, 140, 0), width=3)
        d.text((20, 20), f"f{f}", fill=(255, 0, 80), font=ImageFont.truetype(FD + "Bold.otf", 60))
        sh.paste(fr.resize((tw_, th_), Image.LANCZOS), ((k % 4) * tw_, (k // 4) * th_))
    sh.save(DIR + "contact_sheet.png")
    print("sheet frames", pick)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("scene starts", [round(s, 2) for s in starts], "ends", [round(e, 2) for e in ends])
    if what in ("sheet", "all"): sheet()
    if what in ("audio", "all"): audio()
    if what in ("video", "all"): video()
    if what == "all": mux()
