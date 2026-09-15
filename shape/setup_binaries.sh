#!/bin/bash
# Build convexinv and minkowski on the machine that will run the batch.
# Both need fixes that are NOT in the upstream checkout:
#   * constants.h POINTS_MAX 2000 -> 3000. A TESS visit runs for days, so one session can exceed
#     2000 points and convexinv rejects the WHOLE object ((3550) Link: 2,061). 8000 SEGFAULTS --
#     these are static arrays and the stack cannot hold them.
#   * -std=gnu17. Under C23 an empty parameter list means "no parameters", so the K&R-era source
#     fails to compile on any current gcc.
#   * minkowski.f is shipped but never built. It is the canonical reconstructor and solved a
#     289-facet mesh in 0.79 s that polyhedrec failed at every starting scale after 33 minutes.
set -e
D="$(cd "$(dirname "$0")" && pwd)/DAMIT-convex"
[ -d "$D" ] || { echo "missing $D -- clone mkretlow/DAMIT-convex into shape/"; exit 1; }

cd "$D/convexinv"
grep -q 'POINTS_MAX *3000' constants.h || \
  sed -i.orig 's/#define POINTS_MAX *[0-9]*/#define POINTS_MAX         3000/' constants.h
grep -q 'MAX_N_OBS *20000' constants.h || \
  sed -i 's/#define MAX_N_OBS *[0-9]*/#define MAX_N_OBS         20000/' constants.h
make clean >/dev/null 2>&1 || true
make CFLAGS="-O3 -std=gnu17 -w" >/dev/null
echo "built convexinv"

cd "$D/fortran"
gfortran -O2 ${SDKROOT:+-L$SDKROOT/usr/lib} -o ../minkowski minkowski.f 2>/dev/null \
  || gfortran -O2 -L"$(xcrun --show-sdk-path 2>/dev/null)/usr/lib" -o ../minkowski minkowski.f
echo "built minkowski"

cd "$D/.."
grep -E 'POINTS_MAX|MAX_N_OBS' DAMIT-convex/convexinv/constants.h | head -2
ls -l DAMIT-convex/convexinv/convexinv DAMIT-convex/minkowski | awk '{print "  ",$5,$9}'
