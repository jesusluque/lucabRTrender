#!/bin/zsh
# sparrow-shadow-frame.sh FRAME WIDTHxHEIGHT PATHS OUTDIR
# The four renders a shadow-catcher frame is composited from (docs/decisions.md,
# "A shadow catcher by composition"): A the bird over a white ground, B the
# ground alone, C the bird alone over the sky, M the bird alone with no dome,
# whose alpha is its coverage.
set -e
f=$1; size=$2; paths=$3; out=$4
S=$(cd "$(dirname "$0")" && pwd)
R=/Users/muriel/inn/lucabRTrender/build/macos-arm64-release/bin/lrt
A=/Users/muriel/tools/assets/Sparrow
row=($(sed -n "${f}p" $S/sparrow-film-camera.txt))
cam=(--eye $row[2] $row[3] $row[4] --target $row[5] $row[6] $row[7] --up 0 0 1 --focal $row[8] --size $size --time $f)
$R stage $A/FilmGsWhite.usda  --technique rt --path-total $paths --splat-shadows $cam -o $out/A_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsGround.usda --technique rt --path-total $paths $cam -o $out/B_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsBird.usda   --technique rt --path-total $paths $cam -o $out/C_$f.exr >/dev/null 2>&1
$R stage $A/FilmGsMask.usda   --technique rt --path-total 16 $cam -o $out/M_$f.exr >/dev/null 2>&1
