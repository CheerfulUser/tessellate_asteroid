"""One-off patch of the published asteroid pages: the 'Spin axis: assumed, not fitted' row becomes a
'Pole (ecliptic)' row with the value (as fitted by the orientation scan), and the assumed-pole caveat box
is removed -- what build_pages.py now writes. Poles come from data/poles.csv, extracted from the batch
shape outputs (ozstar: /fred/oz335/rridden/asteroids/shapes/out/*.json, solution.representative_*).
Pages without a shape viewer (lightcurve-only) are left alone. Any page that does not contain exactly
one of each target stops the patch before a file is written.

Usage: python shape/patch_page_pole.py [--dry-run]
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_pages import pole_cell, stat  # noqa: E402

OLD_ROW = '<div class="stat-row"><span class="stat-label">Spin axis</span><span class="stat-value">assumed, not fitted</span></div>'
CAVEAT = re.compile(r'\n  <div class="caveat">\n    The spin axis is ASSUMED, not fitted:.*?</div>', re.S)
KEY = re.compile(r'"key":"([^"]+)"')

poles = {}
with open(f"{ROOT}/data/poles.csv") as fh:
    for r in csv.DictReader(fh):
        poles[r["key"]] = dict(representative_lambda_deg=float(r["lambda_deg"]), representative_beta_deg=float(r["beta_deg"]),
                               pole_fixed=r["pole_fixed"] == "True")
dry = "--dry-run" in sys.argv
pending, bad, skipped = [], [], 0
for path in sorted(glob.glob(f"{ROOT}/asteroid/*.html")):
    s = open(path).read()
    if "viewer.js" not in s:
        skipped += 1
        continue
    k = KEY.search(s).group(1)
    if k not in poles or s.count(OLD_ROW) != 1 or len(CAVEAT.findall(s)) != 1:
        bad.append((path, k in poles, s.count(OLD_ROW), len(CAVEAT.findall(s))))
        continue
    t = s.replace(OLD_ROW, stat("Pole (ecliptic)", pole_cell(poles[k])))
    pending.append((path, CAVEAT.sub("", t)))
print(f"pages to patch: {len(pending)}; lightcurve-only skipped: {skipped}; problems: {len(bad)}")
if bad:
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1)
if not dry:
    for path, t in pending:
        open(path, "w").write(t)
    print("written")
