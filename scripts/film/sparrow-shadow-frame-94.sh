#!/bin/bash
# sparrow-shadow-frame-94.sh FRAME WIDTHxHEIGHT PATHS OUTDIR: sparrow-shadow-frame.sh on the
# 94, through Vulkan (the splat shadows trace inline rays, which CUDA has not).
# The camera rows are read beside the assets there.
set -e
f=$1; size=$2; paths=$3; out=$4
A=/home/ubuntu/tools/assets/Sparrow
R=/home/ubuntu/inn/lucabRTrender/build/linux-x86_64-release/bin/lrt
read -r -a row <<< "$(sed -n "${f}p" $A/film_camera.txt)"
cam=(--eye ${row[1]} ${row[2]} ${row[3]} --target ${row[4]} ${row[5]} ${row[6]} --up 0 0 1 --focal ${row[7]} --size $size --time $f)
export LRT_BACKEND=vulkan
$R stage $A/FilmGsWhite.usda  --technique rt --path-total $paths --splat-shadows "${cam[@]}" -o $out/A_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsGround.usda --technique rt --path-total $paths "${cam[@]}" -o $out/B_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsBird.usda   --technique rt --path-total $paths "${cam[@]}" -o $out/C_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsMask.usda   --technique rt --path-total 16 "${cam[@]}" -o $out/M_$f.exr >/dev/null 2>&1
