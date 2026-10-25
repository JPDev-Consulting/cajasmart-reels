"""Generate one trimmed WAV per scene (am_michael, speed 1.2) + word timings."""
import json, re
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

M = "./models/"
OUT = "./"
# (spoken text, caption text) — caption uses digits so it reads instantly when muted
LINES = [
    ("¿Crees que un café te cuesta solo el café?",
     "¿Crees que un café te cuesta solo el café?"),
    ("Casi todos cuentan solo el café y olvidan lo demás.",
     "Casi todos cuentan solo el café y olvidan lo demás."),
    ("Mira por dentro: café, cuarenta centavos. Leche, treinta. Vaso y tapa, veinticinco.",
     "Mira por dentro: café, $0.40. Leche, $0.30. Vaso y tapa, $0.25."),
    ("Luz, agua y renta, cincuenta. El tiempo de tu equipo, sesenta.",
     "Luz, agua y renta, $0.50. El tiempo de tu equipo, $0.60."),
    ("Total real: dos dólares con cinco centavos por taza.",
     "Total real: $2.05 por taza."),
    ("Si la vendes en dos cincuenta, te quedan cuarenta y cinco centavos.",
     "Si la vendes en $2.50, te quedan $0.45."),
    ("Con CajaSmart controlas tu negocio sin adivinar. ¡Síguenos para más!",
     "Con CajaSmart controlas tu negocio sin adivinar. ¡Síguenos para más!"),
]

k = Kokoro(M + "kokoro-v1.0.onnx", M + "voices-v1.0.bin")
VOICE, SPEED = "ef_dora", 1.1


def phon_len(word):
    try:
        p = k.tokenizer.phonemize(word, "es")
    except Exception:
        p = word
    return max(1, len(re.sub(r"\s", "", p)))


SPOKEN = {"$0.40": "cuarenta centavos", "$0.30": "treinta", "$0.25": "veinticinco", "$0.50": "cincuenta",
          "$0.60": "sesenta", "$2.05": "dos dólares con cinco centavos", "$2.50": "dos cincuenta",
          "$0.45": "cuarenta y cinco centavos"}


def spoken_form(w):
    core = w.strip(".,:!?")
    return SPOKEN.get(core, re.sub(r"[^\wáéíóúñü']", "", core.lower()) or core)


_wd = {}


def word_dur(w):
    """weight = duration of the word synthesized alone with the same voice/speed (better than phoneme counts)."""
    sw = spoken_form(w)
    if sw not in _wd:
        a, sr = k.create(sw, voice=VOICE, speed=SPEED, lang="es")
        i = np.where(np.abs(a) > 0.02 * np.abs(a).max())[0]
        _wd[sw] = (i[-1] - i[0]) / sr if len(i) else 0.1
    return _wd[sw]


def word_times(audio, sr, cap):
    """Phrase boundaries from real pauses in the audio; phoneme-weighted inside phrases."""
    words = cap.split()
    hop = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio) - hop, hop)])
    thr = 0.08 * env.max()
    quiet = env < thr
    # pauses >= 90 ms
    pauses, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= 5 and i > 5 and j < len(quiet) - 5:
                pauses.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    # phrases split at punctuation
    phrases, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.,:!?]$", w):
            phrases.append(cur); cur = []
    if cur:
        phrases.append(cur)
    dur = len(audio) / sr
    wl = lambda ph: sum(word_dur(w) for w in ph)
    nb = len(phrases) - 1
    # expected break = length of the sentence prefix synthesized up to that break (same voice/speed)
    exp, upto = [], []
    for ph in phrases[:-1]:
        upto += ph
        txt = " ".join(spoken_form(w) + (w[-1] if w[-1] in ".,:!?" else "") for w in upto)
        a2, sr2 = k.create(txt, voice=VOICE, speed=SPEED, lang="es")
        ii = np.where(np.abs(a2) > 0.02 * np.abs(a2).max())[0]
        exp.append((ii[-1] - ii[0]) / sr2)
    # global best assignment of breaks to real pauses (ordered), skipping allowed at a cost
    m = len(pauses); INF = 1e9
    best = [[INF] * (m + 1) for _ in range(nb + 1)]; back = [[None] * (m + 1) for _ in range(nb + 1)]
    for j in range(m + 1): best[0][j] = 0.0
    for i in range(1, nb + 1):
        for j in range(m + 1):
            c = best[i - 1][j] + 0.35                     # skip: no pause for this break
            if c < best[i][j]: best[i][j], back[i][j] = c, (i - 1, j, None)
            if j > 0:
                p0, p1 = pauses[j - 1]
                for jj in range(j):                         # previous break used a pause before j
                    cc = best[i - 1][jj] + abs((p0 + p1) / 2 - exp[i - 1]) - 0.4 * min(p1 - p0, 0.25)
                    if cc < best[i][j]: best[i][j], back[i][j] = cc, (i - 1, jj, j - 1)
    j = min(range(m + 1), key=lambda jj: best[nb][jj]); i = nb; pick = [None] * nb
    while i > 0:
        pi, pj, used = back[i][j]; pick[i - 1] = used; i, j = pi, pj
    bounds, t = [], 0.0
    for gi in range(len(phrases)):
        if gi == nb: bounds.append((t, dur)); break
        if pick[gi] is not None:
            p0, p1 = pauses[pick[gi]]; bounds.append((t, p0)); t = p1
        else:
            bounds.append((t, max(t + 0.1, exp[gi]))); t = max(t + 0.1, exp[gi])
    out = []
    for ph, (a, b) in zip(phrases, bounds):
        wts = [word_dur(w) for w in ph]
        tot, acc = sum(wts), a
        for w, wt in zip(ph, wts):
            d = (b - a) * wt / tot
            out.append({"w": w, "s": round(acc, 3), "e": round(acc + d, 3)})
            acc += d
    return out


meta = []
for i, (spoken, cap) in enumerate(LINES):
    a, sr = k.create(spoken, voice=VOICE, speed=SPEED, lang="es")
    thr = 0.02 * np.max(np.abs(a))
    idx = np.where(np.abs(a) > thr)[0]
    a = a[max(0, idx[0] - int(0.02 * sr)): idx[-1] + int(0.06 * sr)]
    sf.write(f"{OUT}v{i}.wav", a, sr)
    d = len(a) / sr
    meta.append({"dur": d, "sr": sr, "caption": cap, "words": word_times(a, sr, cap)})
    print(i, round(d, 2), len(spoken.split()), "words |", spoken)
print("total voice", round(sum(m["dur"] for m in meta), 2), "s")
json.dump(meta, open(OUT + "voice.json", "w"), indent=1)
