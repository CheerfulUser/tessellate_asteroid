"""Patch real per-object viewing geometry into already-built asteroid pages.

The pages were generated with one hardcoded camera for every object (observer at (1,0,0) in the
pole frame, aspect 90). That is wrong on both axes: azimuth about the spin axis is
object-dependent and acts as a constant phase offset -- the wrong face shown at a given
lightcurve phase -- and the measured aspect runs ~67-105 deg, not 90.

Rewrites camQ / sunCam / aspect inside each page's window.AST. Rebuilding instead would need the
batch JSONs pulled back from ozstar; this touches only the three fields that were wrong.

Usage:  python shape/patch_cameras.py [data/cameras.csv]
"""
import csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/cameras.csv'

cams = {}
with open(SRC) as fh:
    for r in csv.DictReader(fh):
        cams[r['key']] = ([float(r['q0']), float(r['q1']), float(r['q2']), float(r['q3'])],
                          [float(r['sx']), float(r['sy']), float(r['sz'])],
                          float(r['aspect']))
print(f'  {len(cams):,} cameras from {os.path.relpath(SRC, ROOT)}')

pat = re.compile(r'(<script>window\.AST=)(\{.*?\})(;</script>)', re.S)
done = missing = unchanged = 0
for path in sorted(glob.glob(f'{ROOT}/asteroid/*.html')):
    key = os.path.basename(path)[:-5]
    cam = cams.get(key)
    if cam is None:
        missing += 1
        continue
    s = open(path).read()
    m = pat.search(s)
    if not m:
        unchanged += 1
        continue
    ast = json.loads(m.group(2))
    ast['D']['camQ'], ast['D']['sunCam'], ast['D']['aspect'] = cam[0], cam[1], cam[2]
    s2 = s[:m.start()] + m.group(1) + json.dumps(ast, separators=(',', ':')) + m.group(3) + s[m.end():]
    open(path, 'w').write(s2)
    done += 1

print(f'  patched {done:,}   no camera row {missing:,}   no AST block {unchanged:,}')
if missing:
    print('  NOTE: pages without a camera row still carry the wrong hardcoded orientation')
