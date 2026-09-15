"""Pages for objects with a rotation period but no shape model.

1,666 of the 16,427 reliable periods produced no convex model -- mostly too few usable points
for inversion, then Minkowski reconstruction failures and meshes rejected for non-convex faces.
Those objects still have a well-determined period and a folded lightcurve, which is the primary
measurement; only the shape is missing. Listing them in the catalogue with no page at all hides
real data.

These pages carry the same information panel and the same folded lightcurve as a full page, and
say plainly why there is no model. No 3D canvas, so they need no mesh and no camera.

Usage:  python shape/build_lc_pages.py
"""
import hashlib, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')
STATUS = os.environ.get('STATUS_CSV', '')
FOLDED = f'{Y34}/folded'
CAT = f'{Y34}/population_figs/all_sector_report_v5_doubled.csv'

WHY = {
    'too few usable points': (
        'Too few usable points for inversion. Convex inversion needs the observer geometry for '
        'every measurement and discards visits shorter than 50 points, since a short fragment '
        'cannot anchor its own brightness scale. A period fit needs neither.'),
    'mesh reconstruction failed': (
        'The fit did not reconstruct into a closed body. Convex inversion solves for facet areas '
        'and normals, and those only assemble into a polyhedron if they satisfy the Minkowski '
        'condition; a weakly constrained fit can have no valid closed solution.'),
    'non-convex faces': (
        'The reconstructed mesh contained non-convex faces and was rejected rather than '
        'published.'),
}
DEFAULT_WHY = 'Shape inversion did not converge on a usable model for this object.'


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


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


def why_text(err):
    e = str(err).strip()
    if re.match(r'only \d+ usable points', e):
        return WHY['too few usable points']
    if re.search(r'\d+ non-convex faces', e):
        return WHY['non-convex faces']
    if 'mesh reconstruction' in e:
        return WHY['mesh reconstruction failed']
    return DEFAULT_WHY


def symmetry_cell(oe):
    if oe is None or not np.isfinite(oe):
        return '&mdash;'
    m = (1.0 - oe) / (1.0 + oe)
    lab = 'monomodal' if m < -0.138 else ('symmetric' if m >= 0.600 else 'asymmetric')
    return f'{m:+.2f} <span class="sym-note">({lab})</span>'


def main():
    c = pd.read_csv(CAT, low_memory=False)
    r = c[(c.is_reliable == True) & c.period_hr.notna()].copy()      # noqa: E712
    r['key'] = r.designation.map(sn)

    have = {f[:-5] for f in os.listdir(f'{ROOT}/asteroid')} if os.path.isdir(f'{ROOT}/asteroid') else set()
    todo = r[~r.key.isin(have)]
    print(f'  {len(r):,} reliable, {len(have):,} with a shape page -> {len(todo):,} to build')

    oe = {}
    try:
        h = pd.read_csv(f'{Y34}/comparison_data/cluster_features.csv')[['designation', 'odd_over_even']]
        oe = {t.designation: float(t.odd_over_even) for t in h.itertuples() if pd.notna(t.odd_over_even)}
    except FileNotFoundError:
        pass

    errs = {}
    if STATUS and os.path.exists(STATUS):
        s = pd.read_csv(STATUS, low_memory=False)
        errs = {t.designation: t.error for t in s.itertuples()}

    os.makedirs(f'{ROOT}/data/lightcurves', exist_ok=True)
    n = skipped = 0
    for t in todo.itertuples():
        fp = f'{FOLDED}/{t.key}.json'
        if not os.path.exists(fp):
            skipped += 1
            continue
        d = json.load(open(fp))
        lc = {'phase': d['phase'], 'flux': d['flux'], 'err': d.get('err')}
        json.dump(lc, open(f'{ROOT}/data/lightcurves/{t.key}.json', 'w'), separators=(',', ':'))

        geo = None if pd.isna(t.albedo_real) else float(t.albedo_real)
        spec = None if pd.isna(t.spec_type) else str(t.spec_type)
        rows = ''.join([
            stat('Taxonomic type', spec if spec else '&mdash;'),
            stat('Diameter', f'{t.diameter_km_real:.1f} km' if pd.notna(t.diameter_km_real) else '&mdash;'),
            stat('Geometric albedo', f'{geo:.3f}' if geo else '&mdash;'),
            stat('Absolute mag <em>H</em>', f'{t.magnitude_H:.2f}' if pd.notna(t.magnitude_H) else '&mdash;'),
            stat('LCDB period', f'{t.published_rot_per_hr:.4f} hours' if pd.notna(t.published_rot_per_hr) else 'none (new)'),
        ])
        sol = ''.join([
            stat('Rotation period', f'{t.period_hr:.5f} hours'),
            stat('Amplitude', f"{d['amplitude_mag']:.3f} mag" if d.get('amplitude_mag') else '&mdash;'),
            stat('Lightcurve symmetry', symmetry_cell(oe.get(t.designation))),
            stat('Observations', f"{d['n_points']:,} pts / {d['n_visits']} visits"),
            stat('Sectors', d.get('n_sectors', '&mdash;')),
        ])
        html = f'''<!doctype html><meta charset="utf-8">
<title>{t.designation} — TESSELLATE asteroid catalog</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="../assets/viewer.css?v={_asset_v("viewer.css")}">
<div class="lconly">
  <a class="home" href="../index.html">&larr; Catalog</a>
  <p class="eyebrow">TESSELLATE rotation period</p>
  <h1>{t.designation}</h1>
  <p class="sec" style="margin-top:0">Physical properties</p>
  {rows}
  <p class="sec">TESSELLATE solution</p>
  {sol}
  <p class="sec">Phase-folded lightcurve</p>
  <canvas id="lc" aria-label="Phase-folded lightcurve of {t.designation}"></canvas>
  <p class="sec">Downloads</p><div class="dl">
    <a href="../data/lightcurves/{t.key}.json" download>Folded lightcurve (JSON)</a></div>
  <div class="caveat">
    <strong>No shape model for this object.</strong> {why_text(errs.get(t.designation))}
    The rotation period and folded lightcurve above are unaffected.
  </div>
</div>
<script>window.LCONLY={json.dumps({'lc': f'../data/lightcurves/{t.key}.json'}, separators=(',', ':'))};</script>
<script src="../assets/lconly.js?v={_asset_v("lconly.js")}"></script>
'''
        open(f'{ROOT}/asteroid/{t.key}.html', 'w').write(html)
        n += 1
    print(f'  built {n:,} lightcurve-only pages   ({skipped} had no folded curve)')


if __name__ == '__main__':
    main()
