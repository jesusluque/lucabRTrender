#!/bin/bash
out=/home/ubuntu/sc/film
mkdir -p $out
for f in $(seq ${1:-206} ${2:-410}); do
    [ -f $out/final_$(printf %03d $f).png ] && continue
    t0=$(date +%s)
    $(dirname "$0")/sparrow-shadow-frame-94.sh $f 1920x1080 64 $out
    oiiotool $out/A_$f.exr --ch R,G,B $out/B_$f.exr --ch R,G,B --div --clamp:min=0:max=1 --subc 1 --mulc 0.589,0.703,0.9025 $out/M_$f.exr --ch A,A,A --mulc -1 --addc 1 --mul $out/C_$f.exr --ch R,G,B --add --colorconvert linear sRGB -o $out/final_$(printf %03d $f).png 2>/dev/null
    rm -f $out/A_$f.exr $out/B_$f.exr $out/C_$f.exr $out/M_$f.exr
    echo "frame $f: $(( $(date +%s) - t0 )) s"
done
echo FILM94_DONE
