#!/bin/zsh
# sparrow-shadow-film.sh [FIRST] [LAST]: every frame of the film as a shadow-catcher
# composite (sky x A/B where the bird is not, the bird where it is), then the mp4
# when the range is the whole. On the Mac; the -94 pair is the same on the 94.
S=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-$HOME/tools/assets/Sparrow/film_shadow}
mkdir -p $out
for f in $(seq ${1:-1} ${2:-410}); do
    [ -f $out/final_$(printf %03d $f).png ] && continue
    t0=$(date +%s)
    $S/sparrow-shadow-frame.sh $f 1920x1080 64 $out
    oiiotool $out/A_$f.exr --ch R,G,B $out/B_$f.exr --ch R,G,B --div --clamp:min=0:max=1 --subc 1 --mulc 0.589,0.703,0.9025 $out/M_$f.exr --ch A,A,A --mulc -1 --addc 1 --mul $out/C_$f.exr --ch R,G,B --add --colorconvert linear sRGB -o $out/final_$(printf %03d $f).png 2>/dev/null
    rm -f $out/A_$f.exr $out/B_$f.exr $out/C_$f.exr $out/M_$f.exr
    echo "frame $f: $(( $(date +%s) - t0 )) s"
done
[ "${2:-410}" = 410 ] && [ "${1:-1}" = 1 ] && ffmpeg -y -framerate 30 -i $out/final_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 16 ~/tools/assets/Sparrow/Film_shadow.mp4 > /dev/null 2>&1
echo FILM_DONE
