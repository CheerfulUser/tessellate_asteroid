"""Generate asteroid pages from the batch shape output.

batch_shapes.py writes one JSON per object holding {solution, mesh, lightcurve}, which is a
different layout from the single-object path (separate *_results.json / *_mesh_data.json fed
through build_shape_viewer.py). This consumes the batch format directly and emits the same
shell + data files, so pages from either path are identical to a visitor.

CAMERA. Read per object from shapes/cameras.csv (built by make_cameras.py), which recovers the
real TESS viewing geometry from each object's per-epoch positions.

An earlier version of this file hardcoded one camera for everything, arguing that a pole fixed
perpendicular to the mean observer makes every object equator-on and therefore identical in body
coordinates. That was wrong twice over. Azimuth about the spin axis is object-dependent and acts
as a constant phase offset, so the viewer showed the wrong face at a given lightcurve phase; and
the aspect is not 90 anyway -- measured across the catalogue it runs from about 67 to 105 deg.
Without a camera row an object is SKIPPED rather than given a plausible-looking default, because
a wrong orientation is worse than a missing page.

Usage:
    python shape/build_pages.py shapes_in/            # all JSONs in a directory
    python shape/build_pages.py shapes_in/ --limit 500
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')
BULK_BASE = os.environ.get('BULK_BASE', '../data/lightcurves')
COORD_DP = 4

CAMERAS = os.environ.get('CAMERAS', os.path.join(os.path.dirname(HERE), 'data', 'cameras.csv'))


def load_cameras():
    """key -> (camQ, sunCam, aspect), the real geometry recovered by make_cameras.py."""
    import csv as _csv
    out = {}
    if not os.path.exists(CAMERAS):
        return out
    with open(CAMERAS) as fh:
        for r in _csv.DictReader(fh):
            out[r['key']] = ([float(r['q0']), float(r['q1']), float(r['q2']), float(r['q3'])],
                             [float(r['sx']), float(r['sy']), float(r['sz'])],
                             float(r['aspect']))
    return out


def _mat2quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


_TINT = {'M': [1.00, 0.93, 0.82], 'S': [1.00, 0.90, 0.76], 'C': [1.00, 0.98, 0.96]}


# Lightcurve symmetry on a signed scale: (even-odd)/(even+odd) = (1-oe)/(1+oe), bounded -1..+1.
# +1 is two identical maxima per rotation, -1 is a single maximum, and ZERO is the physical
# crossover where odd-harmonic power equals even -- so the sign alone says the curve has gone
# single-peaked. Cuts match the catalog tables: monomodal below -0.138 (odd/even 1.32, the
# measured split against the pipeline's own double_peaked flag), symmetric at or above +0.600
# (odd/even 0.25).
def symmetry_cell(p):
    oe = p.get('odd_over_even')
    if oe is None:
        return '&mdash;'
    m = (1.0 - oe) / (1.0 + oe)
    lab = 'monomodal' if m < -0.138 else ('symmetric' if m >= 0.600 else 'asymmetric')
    return f'{m:+.2f} <span class="sym-note">({lab})</span>'


def _asset_v(name):
    """Content hash for a shared asset, so a cached copy is never served
    against a newer page. Assets carry a short TTL, which is long enough for a
    stale stylesheet to look like a missing feature."""
    p = os.path.join(ROOT, "assets", name)
    try:
        return hashlib.sha1(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "0"


def stat(label, value):
    return (f'<div class="stat-row"><span class="stat-label">{label}</span>'
            f'<span class="stat-value">{value}</span></div>')


def build(path, phys, cams):
    d = json.load(open(path))
    s, mesh, lc = d['solution'], d['mesh'], d['lightcurve']
    key = s['key']
    cam = cams.get(key)
    if cam is None:
        raise KeyError(f'no camera row for {key}')
    p = phys.get(s['designation'], {})

    os.makedirs(f'{ROOT}/data/shapes', exist_ok=True)
    os.makedirs(f'{ROOT}/data/lightcurves', exist_ok=True)
    os.makedirs(f'{ROOT}/asteroid', exist_ok=True)
    json.dump(mesh, open(f'{ROOT}/data/shapes/{key}.json', 'w'), separators=(',', ':'))
    json.dump(lc, open(f'{ROOT}/data/lightcurves/{key}.json', 'w'), separators=(',', ':'))

    geo = p.get('albedo')
    spec = p.get('spec')
    rows = ''.join([
        stat('Taxonomic type', spec if spec else '&mdash;'),
        stat('Diameter', f"{p['diam']:.1f} km" if p.get('diam') else '&mdash;'),
        stat('Geometric albedo', f'{geo:.3f}' if geo else '&mdash;'),
        stat('Absolute mag <em>H</em>', f"{p['H']:.2f}" if p.get('H') else '&mdash;'),
        stat('LCDB period', f"{p['lcdb']:.4f} hours" if p.get('lcdb') else 'none (new)'),
    ])
    sol = ''.join([
        stat('Rotation period', f"{s['adopted_period_hr']:.5f} hours"),
        stat('Amplitude', f"{s['amplitude_mag']:.3f} mag" if s.get('amplitude_mag') else '&mdash;'),
        stat('Lightcurve symmetry', symmetry_cell(p)),
        stat('Equatorial ratio', f"{s['equatorial_ratio']:.2f}" if s.get('equatorial_ratio') else '&mdash;'),
        stat('Polar / equatorial', f"{s['polar_ratio']:.2f}" if s.get('polar_ratio') else '&mdash;'),
        stat('Observations', f"{s['n_obs']:,} pts / {s['n_visits']} visits"),
        stat('Baseline', f"{s['baseline_days']:.1f} d"),
        stat('Facets', s.get('n_facets', '&mdash;')),
        stat('Spin axis', 'assumed, not fitted'),
        stat('Viewing aspect', f'{cam[2]:.0f}&deg; from the pole'),
    ])
    dl = ['<p class="sec">Downloads</p><div class="dl">',
          f'<a href="../data/shapes/{key}.json" download>Shape model (JSON mesh)</a>',
          f'<a href="../data/lightcurves/{key}.json" download>Folded lightcurve (JSON)</a>']
    if os.path.exists(f'{ROOT}/data/lightcurves/{key}_stacked.csv.gz'):
        dl.append(f'<a href="{BULK_BASE}/{key}_stacked.csv.gz" download>'
                  f'Full stacked lightcurve (CSV.gz)</a>')
    dl.append('</div>')

    meta = {'key': key, 'has_lc': bool(lc.get('phase')),
            'shape': f'../data/shapes/{key}.json',
            'lc': f'../data/lightcurves/{key}.json',
            'D': {'alb': [1.0] * len(mesh['fn']), 'weak': [False] * len(mesh['fn']),
                  # neutral mid-grey: with the albedo map off every facet takes this one
                  # colour, so it must be a readable surface tone. [0,32,76] is cividis's
                  # DARKEST entry and rendered the body near-black.
                  'lo': 1.0, 'hi': 1.0, 'lut': [[140, 138, 132]],
                  'camQ': cam[0], 'sunCam': cam[1], 'aspect': cam[2],
                  'geoAlbedo': geo if geo else 0.10,
                  'tint': _TINT.get(spec[:1].upper() if spec else '', [1.0, 1.0, 1.0]),
                  'spec': spec if spec else 'unknown'}}

    html = f'''<!doctype html><meta charset="utf-8">
<title>{s['designation']} — TESSELLATE asteroid catalog</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="../assets/viewer.css?v={_asset_v("viewer.css")}">
<canvas id="c" role="img" aria-label="Interactive rotatable 3D convex shape model of asteroid {s['designation']}"></canvas>
<div class="panel info">
  <a class="home" href="../index.html">&larr; Catalog</a>
  <p class="eyebrow">TESSELLATE convex inversion</p>
  <h1>{s['designation']}</h1>
  <p class="sec" style="margin-top:0">Physical properties</p>
  {rows}
  <p class="sec">TESSELLATE solution</p>
  {sol}
  {''.join(dl)}
  <div class="caveat">
    The spin axis is ASSUMED, not fitted: single-apparition data cannot determine a pole, and
    every orientation fits this lightcurve about equally well. The shape is barely affected by
    that choice, but its orientation in space carries no information.
  </div>
</div>
<div class="panel ctrl">
  <div class="row" role="group" aria-label="Rotation mode">
    <button class="btn" id="mLock" aria-pressed="true">Locked to axis</button>
    <button class="btn" id="mFree" aria-pressed="false">Free rotate</button>
    <button class="btn" id="reset">Reset</button>
  </div>
  <div class="row" id="playrow">
    <button class="btn" id="spin" aria-pressed="false">&#9654; Play</button>
    <label for="ph">Phase</label><input type="range" id="ph" min="0" max="1000" value="0">
    <span class="stat-value" id="phv">0.000</span>
  </div>
  <p class="hint" id="hint">Snapped to the TESS viewing geometry. Drag horizontally to turn the
     body about its spin axis &mdash; the lightcurve marker follows. Press Play to rotate at a
     steady rate.</p>
  <canvas id="lc" aria-label="Phase-folded lightcurve with a marker tracking the rotation"></canvas>
</div>
<script>window.AST={json.dumps(meta, separators=(',', ':'))};</script>
<script src="../assets/viewer.js?v={_asset_v("viewer.js")}"></script>
'''
    open(f'{ROOT}/asteroid/{key}.html', 'w').write(html)
    return key


def load_phys():
    import pandas as pd
    c = pd.read_csv(f'{Y34}/population_figs/all_sector_report_v5_doubled.csv', low_memory=False)
    out = {}
    for t in c.itertuples():
        out[t.designation] = dict(
            diam=None if pd.isna(t.diameter_km_real) else float(t.diameter_km_real),
            albedo=None if pd.isna(t.albedo_real) else float(t.albedo_real),
            spec=None if pd.isna(t.spec_type) else str(t.spec_type),
            H=None if pd.isna(t.magnitude_H) else float(t.magnitude_H),
            lcdb=None if pd.isna(t.published_rot_per_hr) else float(t.published_rot_per_hr))
    try:
        h = pd.read_csv(f'{Y34}/comparison_data/cluster_features.csv')[
            ['designation', 'odd_over_even']]
        for t in h.itertuples():
            if t.designation in out and pd.notna(t.odd_over_even):
                out[t.designation]['odd_over_even'] = float(t.odd_over_even)
    except FileNotFoundError:
        pass
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('indir')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    files = sorted(glob.glob(f'{a.indir}/*.json'))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    if a.limit:
        files = files[:a.limit]
    phys = load_phys()
    cams = load_cameras()
    print(f'{len(files):,} shape files, {len(cams):,} cameras')
    if not cams:
        sys.exit(f'no cameras at {CAMERAS} -- run make_cameras.py first')
    n = 0
    for f in files:
        try:
            build(f, phys, cams)
            n += 1
        except Exception as e:
            print(f'  {os.path.basename(f)}: {type(e).__name__}: {e}')
    print(f'built {n:,} pages in {ROOT}/asteroid/')
