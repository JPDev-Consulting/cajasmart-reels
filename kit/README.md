# CajaSmart Reel kit

Reference code used to produce the CajaSmart Reels. Scheduled runs copy these files from the project into their workspace.

## Files
- `make_logo.py` — rebuilds the CajaSmart logo (traced from cajasmart.com/CajaSmartLogo.png) as `logo.png`, white on transparent, 2024x852. Run it first.
- `voice.py` — Kokoro TTS: one trimmed WAV per scene (ef_dora, speed 1.1, es) plus word timings in `voice.json`. Edit `LINES` (spoken text, caption text) and `SPOKEN` (how $/% tokens are said).
- `render_brand_style.py` — style (a): navy #252836 background, headline plus animated drawing per scene, captions near the bottom. Replace the `art0`..`art6` functions, `HEAD` and the `beats` dict for a new topic.
- `render_explainer_style.py` — style (b): near-black grid, continuous camera, cross-section world, gold arcs, callouts, stat card, top captions with selection boxes, follow sheet. Replace `world()`/`overlay()` content, `CAM` and `KEYS` for a new topic.
- `finish.sh <output.mp4>` — renders audio and video with `render.py`, applies two-pass loudnorm, verifies with ffprobe and writes `contact_sheet_final.png` from the encoded file.

## Setup in a fresh workspace
```bash
mkdir -p models && cd models
wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd .. && pip install --break-system-packages -q kokoro-onnx soundfile
python3 make_logo.py
# edit voice.py LINES, then:
python3 voice.py
cp render_<style>.py render.py   # then replace the scene functions
python3 render.py sheet           # check contact_sheet.png, fix, repeat
bash finish.sh cajasmart_<slug>_ES.mp4
```
