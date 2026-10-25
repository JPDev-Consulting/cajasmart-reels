#!/bin/bash
# full render + two-pass loudnorm + contact sheet from the encoded file
set -e; cd "$(dirname "$0")"; O=$1
python3 render.py audio >/dev/null; python3 render.py video | tail -1
J=$(ffmpeg -hide_banner -i mix.wav -af loudnorm=I=-14:TP=-2:LRA=11:print_format=json -f null - 2>&1 | sed -n '/{/,/}/p')
g(){ echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -y -loglevel error -i video_only.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -af "loudnorm=I=-14:TP=-2:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true" -ar 48000 -c:a aac -b:a 192k -t 30 -movflags +faststart $O
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration -of compact $O
ffmpeg -hide_banner -i $O -af ebur128=peak=true -f null - 2>&1 | grep -E " I:| Peak:" | tail -2
python3 - "$O" <<'P'
import subprocess, sys
from PIL import Image, ImageDraw
import render as r
pick=[0]+[round((r.starts[i]+min(r.ends[i]-r.starts[i]-0.15, r.OFF[i]+r.VM[i]["dur"]))*30) for i in range(7)]
pick[-1]=round((r.starts[6]+5.0)*30)
sh=Image.new("RGB",(360*4,640*2))
for k,f in enumerate(pick):
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",sys.argv[1],"-vf",f"select=eq(n\\,{f})","-vframes","1",f"cs{k}.png"],check=True)
    im=Image.open(f"cs{k}.png").convert("RGB"); d=ImageDraw.Draw(im)
    for y in (250,1450): d.line([(0,y),(1080,y)],fill=(255,0,80),width=4)
    sh.paste(im.resize((360,640),Image.LANCZOS),((k%4)*360,(k//4)*640))
sh.save("contact_sheet_final.png")
P
