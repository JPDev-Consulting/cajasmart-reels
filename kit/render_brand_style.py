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


# ---- scene art (local coords, box 870x650) ----
AX, AY, AW, AH = 90, 560, 870, 650


def coin_path(i, t, x0, y0, spread=1.0):
    """deterministic flying coin: index i, time t -> (x, y, alive)."""
    ph = (t * 0.9 + i * 0.37) % 1.0
    ang = -1.9 + 0.5 * math.sin(i * 2.3) * spread + (i % 2) * 0.9 * spread
    v = 520 + 90 * math.sin(i * 1.7)
    x = x0 + math.cos(ang) * v * ph
    y = y0 + math.sin(ang) * v * ph + 380 * ph * ph
    return x, y, ph


def icon_drawer(p, cx, cy, s=1.0, open_=1.0):
    p.rr(cx - 210 * s, cy - 40 * s, cx + 210 * s, cy + 120 * s, 22 * s, fill=(64, 70, 94))       # body
    p.rr(cx - 150 * s, cy - 150 * s, cx + 150 * s, cy - 50 * s, 16 * s, fill=(76, 84, 110))     # screen
    p.rr(cx - 130 * s, cy - 135 * s, cx + 130 * s, cy - 65 * s, 10 * s, fill=DARK)
    dy = 70 * s * open_
    p.rr(cx - 190 * s, cy + 60 * s + dy, cx + 190 * s, cy + 130 * s + dy, 14 * s, fill=(90, 98, 128))
    for k in range(4):
        x = cx - 140 * s + k * 93 * s
        p.rr(x, cy + 70 * s + dy, x + 75 * s, cy + 100 * s + dy, 6 * s, fill=GREEN if k % 2 == 0 else GREEN_L)


def account(p, x0, y0, x1, y1, col, num, title, sub="", k=1.0, tcol=TXT):
    if k <= 0: return
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * k, (y1 - y0) * k
    p.rr(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 22 * k, fill=CARD, outline=col, w=5)
    if k > 0.8:
        node(p, x0 + 52, cy, 32, col, num, DARK if col == YEL else TXT, 36)
        p.tx(x0 + 100, cy - (16 if sub else 0), title, 40, tcol, "Black", anchor="lm")
        if sub: p.tx(x0 + 100, cy + 30, sub, 28, MUTED, "Bold", anchor="lm")


def art0(p, t, T):  # HOOK: you sell well... but where does the money go?
    tq = wt(0, "nunca") - 0.05
    icon_drawer(p, 435, 420, 1.0, open_=0.6 + 0.4 * eoc(t / 0.3))
    for i in range(9):                                     # coins already flying on frame 0
        x, y, ph = coin_path(i, t + 0.4, 435, 480)
        if y < 600 and ph < 0.95: icon_coin(p, x, y, 30 * (1 - 0.4 * ph))
    q = eob((t - tq) / 0.35, 2.0)
    if q > 0:
        for j, (x, y) in enumerate(((150, 110), (720, 90), (600, 190))):
            k = eob((t - tq - j * 0.12) / 0.3)
            if k > 0: node(p, x, y, 46 * k, YEL, "?", DARK, 64 * k)


def art1(p, t, T):  # PROBLEM: everything falls into one account and mixes
    tm = wt(1, "mezcla") - 0.1
    mixd = t >= tm
    shake = 9 * math.sin((t - tm) * 55) * math.exp(-(t - tm) * 6) if mixd else 0
    x0, y0, x1, y1 = 185 + shake, 300, 685 + shake, 600
    for i in range(10):                                    # money in (green) and costs (coral) dropping in
        ph = (t * 0.8 + i * 0.13) % 1.0
        col = GREEN_L if i % 2 == 0 else CORAL
        x = 250 + (i * 53) % 370 + shake; y = 20 + 330 * ph
        if mixd: x += 25 * math.sin(t * 9 + i)
        icon_coin(p, x, y, 26, col)
    p.rr(x0, y0, x1, y1, 30, fill=CARD, outline=CORAL if mixd else MUTED, w=6)
    p.tx((x0 + x1) / 2, y0 + 60, "UNA SOLA CUENTA", 40, TXT, "Black")
    for i in range(12):
        col = GREEN_L if (i * 7) % 3 else CORAL
        cx = x0 + 70 + (i % 6) * 72 + (8 * math.sin(t * 10 + i) if mixd else 0)
        cy = y0 + 150 + (i // 6) * 80
        icon_coin(p, cx, cy, 26, col)
    if mixd:
        pill(p, (x0 + x1) / 2, y1 + 10, "todo mezclado", 32 * eob((t - tm) / 0.3), CORAL, DARK)


def art2(p, t, T):  # SOLUTION: split your money into 4 accounts
    t4 = wt(2, "4") - 0.15
    k = eob(t / 0.35)
    p.rr(235, 20, 635, 140, 26, fill=CARD, outline=MUTED, w=5)
    if k > 0.5:
        icon_coin(p, 300, 80, 32); p.tx(460, 80, "TU DINERO", 40, TXT, "Black")
    cols = (BLUE, YEL, GREEN_L, MUTED)
    labs = ("VENTAS", "IMPUESTOS", "GANANCIA", "GASTOS")
    for i in range(4):
        cx = 115 + i * 213
        a = (t - t4 + 0.25 - i * 0.08) / 0.35
        seg(p, (435, 145), (cx, 330), a, cols[i], 7, arrow=True)
        b = eob((t - t4 - i * 0.12) / 0.3)
        if b > 0:
            p.rr(cx - 95 * b, 350, cx + 95 * b, 350 + 230 * b, 22, fill=CARD, outline=cols[i], w=5)
            if b > 0.8:
                node(p, cx, 420, 38, cols[i], str(i + 1), DARK if i in (1, 2) else TXT, 44)
                p.tx(cx, 505, labs[i], 24 if i == 1 else 27, TXT, "Black")


def art3(p, t, T):  # ITEMS 1-2: sales (all money in) / taxes (no surprises)
    t2 = wt(3, "Dos") - 0.1
    k1 = eoc(t / 0.35)
    account(p, 20 + 50 * (1 - k1), 30, 850, 250, BLUE, "1", "Ventas", "aquí entra todo", k1)
    if k1 > 0.8:
        for i in range(4):
            ph = (t * 1.1 + i * 0.25) % 1.0
            icon_coin(p, 640 + 60 * i - 120 * ph, 80 + 110 * ph, 22, GREEN_L)
    k2 = eoc((t - t2) / 0.35)
    if k2 > 0:
        account(p, 20 + 50 * (1 - k2), 330, 850, 550, YEL, "2", "Impuestos", "sin sorpresas", k2)
        if k2 > 0.8:
            lock(p, 740, 440, 70 * eob((t - t2 - 0.3) / 0.35), YEL)


def art4(p, t, T):  # ITEMS 3-4: your profit (first, for you) / expenses (what's left)
    t4 = wt(4, "Cuatro") - 0.1
    k1 = eoc(t / 0.35)
    account(p, 20 + 50 * (1 - k1), 30, 850, 250, GREEN_L, "3", "Tu ganancia", "primero para ti", k1)
    if k1 > 0.8:
        person(p, 710, 225, 0.55, SKIN[0], apron=GREEN, arm=0.6 + 0.2 * math.sin(t * 6))
        check(p, 800, 70, 26, (t - 0.5) / 0.35)
    k2 = eoc((t - t4) / 0.35)
    if k2 > 0:
        account(p, 20 + 50 * (1 - k2), 330, 850, 550, MUTED, "4", "Gastos", "con lo que queda", k2)
        if k2 > 0.8:
            for i, lab in enumerate(("luz", "renta", "café")):
                q = eob((t - t4 - 0.3 - i * 0.12) / 0.3)
                if q > 0: pill(p, 545 + i * 105, 440, lab, 28 * q, TRACK, TXT, padx=14)


def art5(p, t, T):  # TWIST: $10 of every $100 for you, before paying anything
    t10, ta = wt(5, "$10") - 0.1, wt(5, "antes") - 0.1
    g = eoc(t / 0.4)
    x0, x1, y0, y1 = 40, 830, 120, 240
    full = (x1 - x0) * g
    p.tx(40, 70, f"${100 * g:.0f} que vendes", 40, TXT, "Black", anchor="lm")
    s = eoc((t - t10) / 0.5)                   # the $10 slice slides out
    w10 = (x1 - x0) * 0.10
    p.rr(x0 + w10 + 8, y0, x0 + full, y1, 20, fill=BLUE)
    if g > 0.7: p.tx((x0 + w10 + x1) / 2 + 4, (y0 + y1) / 2, "$90 impuestos y gastos", 34, TXT, "Black")
    sx, sy = x0, y0 + 260 * s
    p.rr(sx, sy, sx + w10 + 160 * s, sy + (y1 - y0), 20, fill=GREEN_L)
    p.tx(sx + (w10 + 160 * s) / 2, sy + 60, "$10", 40, DARK, "Black")
    if s > 0.9:
        p.tx(sx + w10 + 190, sy + 60, "para ti", 44, GREEN_L, "Black", anchor="lm")
    q = eob((t - ta) / 0.35)
    if q > 0:
        pill(p, 640, 450, "1º", 52 * q, YEL, DARK, padx=30)
        check(p, 760, 450, 34, (t - ta - 0.2) / 0.35)
    for i in range(3):                          # coins dropping into "your" slice
        c = (t - t10 - 0.5 - i * 0.2)
        if c > 0:
            ph = min(1, c / 0.5)
            if ph < 1: icon_coin(p, 110 + i * 40, 300 + 150 * ph, 22)


def art6(p, t, T):  # CTA: CajaSmart shows today's sales instantly. Follow us!
    k = eob(t / 0.4)
    x0, y0, x1, y1 = 40, 20, 560, 410
    p.rr(x0, y0, x1, y1, 36 * k, fill=DARK)
    p.rr(x0 + 18, y0 + 18, x1 - 18, y1 - 18, 24, fill=(43, 47, 64))
    logo(p, 76, 72, 40, anchor="lm")
    if int(t * 2.5) % 2 == 0: p.el(388, 72, 9, fill=GREEN_L)
    p.tx(404, 72, "EN VIVO", 26, GREEN_L, "Black", anchor="lm")
    p.tx(80, 135, "Ventas de hoy", 30, MUTED, "Bold", anchor="lm")
    q = eoc((t - 0.3) / 1.6)
    p.tx(80, 205, f"${1250 * q:,.0f}", 84, GREEN_L, "Black", anchor="lm")
    for i in range(7):                          # hourly bars growing
        h = (40 + 90 * abs(math.sin(i * 1.4 + 0.5))) * eoc((t - 0.4 - i * 0.08) / 0.4)
        p.rr(80 + i * 62, 370 - h, 80 + i * 62 + 44, 370, 8, fill=BLUE if i != 4 else GREEN_L)
    robot(p, 720, 400, 0.95 * eob((t - 0.25) / 0.4), wave=t * 6, blink=int(t * 10) % 25 == 0)
    lg = eob((t - 0.15) / 0.4)
    if lg > 0.05: logo(p, 435, 470, 98 * lg, anchor="mm")
    tf = wt(6, "¡Sí") - 0.1
    b = eob((t - tf) / 0.35)
    by = 592
    if b > 0:
        s = b * (1 + 0.04 * math.sin(max(0, t - tf - .4) * 6))
        lab = "¡Síguenos!"
        w = (tw(lab, 54, "Black") + 120) * s; h = 104 * s
        p.rr(435 - w / 2, by - h / 2, 435 + w / 2, by + h / 2, h / 2, fill=GREEN)
        p.tx(435, by + 1, lab, 54 * s, TXT, "Black")
        if t - tf > 0.6:  # glow ring expanding around the button (never over the text)
            r = ((t - tf - 0.6) % 1.4) / 1.4; e = 6 + 26 * r
            p.rr(435 - w / 2 - e, by - h / 2 - e, 435 + w / 2 + e, by + h / 2 + e, h / 2 + e,
                 outline=mix(GREEN_L, BG, 1 - r), w=4)


ART = [art0, art1, art2, art3, art4, art5, art6]

HEAD = [
    [[("¿Vendes bien", TXT)], [("y no ves el dinero?", YEL)]],
    [[("Todo en una", TXT)], [("sola cuenta", CORAL)]],
    [[("La solución:", TXT)], [("4 cuentas", GREEN_L)]],
    [[("1. Ventas", BLUE)], [("2. Impuestos", YEL)]],
    [[("3. Tu ganancia", GREEN_L)], [("4. Gastos", MUTED)]],
    [[("Primero tú:", TXT)], [("$10 de cada $100", GREEN_L)]],
    [[("Mira tus ventas", TXT)], [("al instante", GREEN_L)]],
]

# ---- caption pages (1–3 words, break at punctuation) ----
PAGES = []
for sc, m in enumerate(VM):
    pages, cur = [], []
    for w in m["words"]:
        cur.append(w)
        if len(cur) == 3 or w["w"][-1] in ".,:!?":
            pages.append(cur); cur = []
    if cur: pages.append(cur)
    PAGES.append(pages)


def draw_bg(p, f):
    t = f / FPS
    for j in range(9):  # brand wave lines (like the site hero)
        pts = []
        for i in range(0, 1101, 50):
            y = 1560 + j * 26 + 70 * math.sin(i / 260 + t * 0.6 + j * 0.35) + 40 * math.sin(i / 120 - t * 0.4)
            pts.append((i - 10, y))
        p.ln(pts, WAVE, 2, cap=False)
    for j in range(6):
        pts = [(i - 10, 140 + j * 22 + 50 * math.sin(i / 300 - t * 0.5 + j * 0.4)) for i in range(0, 1101, 60)]
        p.ln(pts, WAVE, 2, cap=False)


def draw_chrome(p, sc, ls):
    logo(p, 92, 282, 58)
    for i in range(7):  # progress segments
        x = 735 + i * 33
        col = GREEN_L if i == sc else (MUTED if i < sc else TRACK)
        p.rr(x, 274, x + 26, 286, 6, fill=col)


def draw_head(p, sc, ls):
    k = 1.0 if sc == 0 else eoc(ls / 0.22)
    pop = (0.93 + 0.07 * eob(ls / 0.25)) if sc == 0 else 1.0
    size = 84 * pop
    for li, line in enumerate(HEAD[sc]):
        y = 395 + li * 98 + (1 - k) * 30
        fs = min(size, size * 820 / sum(tw(s, size, "Black") for s, _ in line))
        total = sum(tw(s, fs, "Black") for s, _ in line)
        x = 525 - total / 2
        for s, col in line:
            p.tx(x, y, s, fs, mix(col, BG, k), "Black", anchor="lm")
            x += tw(s, fs, "Black")


def draw_caption(p, sc, ls):
    vt = ls - OFF[sc]
    pages = PAGES[sc]
    cur = None
    for i, pg in enumerate(pages):
        nxt = pages[i + 1][0]["s"] if i + 1 < len(pages) else 99
        if pg[0]["s"] - 0.05 <= vt < nxt - 0.05: cur = pg
    if cur is None:
        if vt > pages[-1][-1]["e"]: cur = pages[-1]  # hold last page
        else: return
    size = 62
    words = [w["w"] for w in cur]
    widths = [tw(w, size, "Black") for w in words]
    space = tw(" ", size, "Black")
    total = sum(widths) + space * (len(words) - 1)
    x = 525 - total / 2
    for w, ww, info in zip(words, widths, cur):
        dt = vt - info["s"] + 0.05
        if dt >= 0:
            k = eob(dt / 0.18, 2.2)
            active = info["s"] - 0.05 <= vt < info["e"] + 0.05 or (info is cur[-1] and vt >= info["s"])
            col = YEL if active else TXT
            p.tx(x + ww / 2, 1300, w, size * (0.6 + 0.4 * k), col, "Black", stroke=6, sfill=DARK)
        x += ww + space


HOOK0 = None


def render_frame(f):
    t = f / FPS
    sc = max(i for i in range(7) if starts[i] <= t + 1e-6)
    ls = t - starts[sc]; T = ends[sc] - starts[sc]
    img = Image.new("RGB", (W * S, H * S), BG)
    p = Pen(img)
    draw_bg(p, f)
    draw_chrome(p, sc, ls)
    draw_head(p, sc, ls)
    layer = Image.new("RGBA", (AW * S, AH * S), (0, 0, 0, 0))
    ART[sc](Pen(layer), ls, T)
    # punch-in on every cut + on the scene's key beat
    beats = {0: [0.0, wt(0, "nunca")], 1: [0.0, wt(1, "mezcla")], 2: [0.0, wt(2, "4")], 3: [0.0, wt(3, "Dos")],
             4: [0.0, wt(4, "Cuatro")], 5: [0.0, wt(5, "antes")], 6: [0.0, wt(6, "¡Sí")]}[sc]
    z = 1.0
    for b in beats:
        if ls >= b: z = punch(ls - b, 0.085 if b == 0 else 0.05)
    if sc == 0 and ls < 1.0: z = 1.0  # hook frame: strongest visual already settled on frame 0
    if abs(z - 1) > 1e-3:
        nw, nh = int(AW * S * z), int(AH * S * z)
        lay = layer.resize((nw, nh), Image.BILINEAR)
        img.paste(lay, ((AX * S) - (nw - AW * S) // 2, (AY * S) - (nh - AH * S) // 2), lay)
    else:
        img.paste(layer, (AX * S, AY * S), layer)
    draw_caption(p, sc, ls)
    out = img.reduce(S)
    return out


def video():
    global HOOK0
    HOOK0 = render_frame(0)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "17", "-pix_fmt", "yuv420p", DIR + "video_only.mp4"], stdin=subprocess.PIPE)
    for f in range(N):
        fr = render_frame(f)
        tail = N - 1 - f
        if tail < 8:  # seamless loop seam: ease the last 8 frames back to the hook frame
            fr = Image.blend(fr, HOOK0, (8 - tail) / 8 * 0.85)
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
                    DIR + "cajasmart_ganancia_ES.mp4"], check=True)


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
