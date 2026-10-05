"""One-off patch of the published asteroid pages for the km scale bar (assets/viewer.js).

Inserts the catalogue diameter as window.AST.diam_km -- what build_pages.py now writes for new
builds, from the same source (load_phys) -- and bumps the viewer.js cache-busting hash, without a
full rebuild (the batch shape JSONs build_pages.py consumes are not kept in this repo). Each
inserted value is checked against the page's own displayed Diameter (rounded to 0.1 km); any
mismatch stops the patch before a file is written.

The viewer falls back to the displayed Diameter when diam_km is absent, but that is rounded to
0.1 km, which would misscale sub-km objects by 10% or more.

Usage: python shape/patch_page_diameters.py [--dry-run]
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_pages import load_phys, _asset_v  # noqa: E402
from batch_shapes import sn  # noqa: E402

AST_RE = re.compile(r"<script>window\.AST=(\{.*?\});</script>", re.S)
DIAM_RE = re.compile(r'<span class="stat-label">Diameter</span><span class="stat-value">([^<]*)</span>')
JS_RE = re.compile(r'viewer\.js\?v=[0-9a-f]+')

dry = "--dry-run" in sys.argv
phys = load_phys()
by_key = {sn(d): v.get("diam") for d, v in phys.items()}
new_v = _asset_v("viewer.js")
patched, no_diam, skipped, bad = [], 0, 0, []
pending = []
for path in sorted(glob.glob(f"{ROOT}/asteroid/*.html")):
    s = open(path).read()
    m = AST_RE.search(s)
    if not m or "viewer.js" not in s:
        skipped += 1          # lightcurve-only pages (lconly.js) have no shape viewer
        continue
    ast = json.loads(m.group(1))
    d = by_key.get(ast["key"])
    shown = DIAM_RE.search(s)
    shown = shown.group(1) if shown else None
    if d is None:
        no_diam += 1
        if shown and "km" in shown:
            bad.append((path, "page shows a diameter the catalogue lacks", shown))
    elif shown != f"{d:.1f} km":
        bad.append((path, f"catalogue {d:.4f} km", shown))
    ast["diam_km"] = None if d is None else round(d, 4)
    t = s[:m.start(1)] + json.dumps(ast, separators=(",", ":")) + s[m.end(1):]
    t = JS_RE.sub(f"viewer.js?v={new_v}", t)
    pending.append((path, t))
    patched.append(path)

print(f"pages with a viewer: {len(patched)}; without a catalogue diameter: {no_diam}; lightcurve-only skipped: {skipped}")
print(f"viewer.js hash -> {new_v}")
if bad:
    print(f"{len(bad)} mismatches, nothing written:")
    for b in bad[:20]:
        print("  ", *b)
    sys.exit(1)
if not dry:
    for path, t in pending:
        open(path, "w").write(t)
    print("written")
