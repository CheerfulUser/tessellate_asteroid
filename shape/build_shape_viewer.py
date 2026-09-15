"""Build the interactive shape viewer: rotatable, spinnable about the recovered pole, coloured
by the albedo map, with a phase-folded lightcurve tracker.

Usage:  python build_shape_viewer.py eurydike

Reads {tag}_mesh_data.json, {tag}_results.json and (optionally) {tag}_albedo.json.

The spin is about the BODY z-axis, which is the spin axis by convexinv's own convention -- the
shape is solved in a frame where rotation is about z, so "rotate about the determined axis" is
exactly a rotation about mesh z. The lightcurve marker is synchronised to that same angle:
viewer phase p maps to a body rotation of 2*pi*p, and the folded curve is binned on the SAME
phase convention (phi = phi0 + 2pi(t-t0)/P), so the marker genuinely tracks which face is
pointing at you rather than being a decorative animation.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
Y34_CAT = ('/Users/rridden/Documents/work/code/tess/asteroid/y3_4/'
           'population_figs/all_sector_report_v4.csv')
from real_shape_tess import TARGETS, prepare  # noqa: E402

# Per-facet albedo mapping is DISABLED. The inversion improved the fit by only 0.6% over a
# uniform surface, with 95 of 289 zones effectively unobserved -- a permissible albedo map, not
# a detected one, and shape/albedo are degenerate in disc-integrated photometry anyway. The
# published UNIFORM geometric albedo is kept: that is a real catalogue measurement and is what
# sets the true-colour brightness.
USE_ALBEDO_MAP = False

TAG = sys.argv[1] if len(sys.argv) > 1 else 'eurydike'
C = TARGETS[TAG]
mesh = json.load(open(os.path.join(HERE, f'{TAG}_mesh_data.json')))
res = json.load(open(os.path.join(HERE, f'{TAG}_results.json')))
alb_path = os.path.join(HERE, f'{TAG}_albedo.json')
ALB = json.load(open(alb_path)) if (USE_ALBEDO_MAP and os.path.exists(alb_path)) else None

verts = np.array(mesh['recovered']['verts'])
facets = mesh['recovered']['facets']

# ---- per-face albedo, matched by normal direction (the mesh tessellation and the Gaussian
# image are the same 33 directions, but the ordering is not guaranteed to survive reconstruction)
face_alb, face_weak = [], []
if ALB:
    an = np.array(ALB['normals']); av = np.array(ALB['albedo']); aw = np.array(ALB['weak'])
    for f in facets:
        p = verts[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        n = n / nn if nn > 0 else np.array([0, 0, 1.0])
        if np.dot(n, p.mean(axis=0)) < 0:
            n = -n
        j = int(np.argmax(an @ n))
        face_alb.append(float(av[j])); face_weak.append(bool(aw[j]))
    lo, hi = float(np.min(face_alb)), float(np.max(face_alb))
else:
    face_alb = [1.0] * len(facets); face_weak = [False] * len(facets); lo, hi = 1.0, 1.0

# ---- folded lightcurve on the SAME phase convention the spin uses
bi = int(np.argmin([abs(s['lambda_deg'] - res['representative_lambda_deg'])
                    + abs(s['beta_deg'] - res['representative_beta_deg'])
                    for s in res['all_solutions']]))
pl = open(os.path.join(HERE, f'{TAG}_params_{bi}.txt')).read().split()
per_hr, t0_jd, phi0 = float(pl[2]), float(pl[3]), float(pl[4])
df = prepare(pd.read_csv(C['lc']))
jd = df['mjd'].values + 2400000.5
phase = ((phi0 + 2 * np.pi * (jd - t0_jd) / (per_hr / 24.0)) / (2 * np.pi)) % 1.0
fl = df['rel_flux'].values
NB = 72
b = np.clip((phase * NB).astype(int), 0, NB - 1)
cnt = np.bincount(b, minlength=NB)
sm = np.bincount(b, weights=fl, minlength=NB)
ok = cnt >= 3
lc_phase = ((np.arange(NB) + 0.5) / NB)[ok].tolist()
lc_flux = (sm[ok] / cnt[ok]).tolist()

# standard error per bin -- a binned curve without it hides how well each point is determined,
# and the per-bin population varies by an order of magnitude across the fold
_mean = sm / np.maximum(cnt, 1)
_var = np.bincount(b, weights=(fl - _mean[b]) ** 2, minlength=NB)
with np.errstate(divide='ignore', invalid='ignore'):
    _sem = np.sqrt(_var / np.maximum(cnt - 1, 1)) / np.sqrt(np.maximum(cnt, 1))
lc_err = np.nan_to_num(_sem[ok]).tolist()

# Peak-to-peak amplitude of the BINNED fold, in magnitudes. Binned rather than raw because the
# raw extremes are set by the noisiest single points; magnitudes because that is the convention
# the LCDB reports amplitudes in, so the two are directly comparable.
_lf = np.array(lc_flux)
AMP_MAG = float(-2.5 * np.log10(_lf.min() / _lf.max())) if len(_lf) and _lf.min() > 0 else float('nan')

# convexinv's OWN fit, folded on the identical grid. Using the fit file rather than
# re-simulating from the mesh guarantees the overlay is the model actually fitted to these
# points -- same scattering law, phase function and per-session scaling.
lc_model = None
try:
    _mod = np.loadtxt(os.path.join(HERE, f'{TAG}_fit_{bi}.txt'))
    if len(_mod) == len(df):
        # Rescale EACH SESSION to its own data before binning. convexinv fits relative
        # photometry with a free brightness scale per session, so its model values are not on a
        # common scale -- for (3550) Link the model/data ratio runs 0.886 to 1.233 across seven
        # sessions. A single global renormalisation leaves that 39% spread in place, and since
        # each session covers its own span of rotational phase, the mismatch appears as regular
        # spikes in the folded model and inflates its amplitude (0.505 mag against the data's
        # 0.291). The data is already per-visit normalised; the model has to be too.
        _mod = _mod.astype(float).copy()
        for _v in pd.unique(df['visit'].values):
            _m = (df['visit'].values == _v)
            _dm, _mm = df['rel_flux'].values[_m].mean(), _mod[_m].mean()
            if np.isfinite(_mm) and _mm != 0:
                _mod[_m] *= _dm / _mm
        _ms = np.bincount(b, weights=_mod, minlength=NB) / np.maximum(cnt, 1)
        _mv = _ms[ok]
        lc_model = [round(float(x), 6) for x in _mv]
    else:
        print(f'  fit file has {len(_mod)} rows vs {len(df)} obs -- model overlay skipped')
except Exception as e:
    print(f'  model overlay unavailable ({e})')

# ---- the ACTUAL TESS viewing geometry, expressed in the pole frame (spin phase removed).
# convexinv's transform is v_body = Rz(phi) Ry(pi/2-beta) Rz(lambda) v_inertial, so dropping the
# Rz(phi) leaves a frame whose z is the spin axis and in which the observer sits at a nearly
# fixed direction -- the aspect changes by well under a degree over these observations. Snapping
# the camera there shows the body as TESS actually saw it.
from real_shape_tess import build_geometry  # noqa: E402
_sv, _ev = build_geometry(df)
_en = _ev / np.linalg.norm(_ev, axis=1, keepdims=True)
_sn = _sv / np.linalg.norm(_sv, axis=1, keepdims=True)
_l, _b = np.radians(res['representative_lambda_deg']), np.radians(res['representative_beta_deg'])
_Rz = np.array([[np.cos(_l), np.sin(_l), 0], [-np.sin(_l), np.cos(_l), 0], [0, 0, 1]])
_a = np.pi / 2 - _b
_Ry = np.array([[np.cos(_a), 0, -np.sin(_a)], [0, 1, 0], [np.sin(_a), 0, np.cos(_a)]])
_Rpf = _Ry @ _Rz
_epf = (_Rpf @ _en.T).T.mean(axis=0); _epf /= np.linalg.norm(_epf)
_spf = (_Rpf @ _sn.T).T.mean(axis=0); _spf /= np.linalg.norm(_spf)
# SIGNED aspect: arccos of the signed z-component, not its absolute value. Using abs() threw
# away which hemisphere TESS observes from, and they are different geometries -- for all four
# published objects the observer sits at NEGATIVE z (e_pf_z = -0.44 to -0.63), so an "aspect of
# 53 deg" actually means 127 deg from the +z pole. That sign error made a facet at latitude
# -84.8 deg look hidden when it is in fact the face pointing straight at TESS.
ASPECT = float(np.degrees(np.arccos(np.clip(_epf[2], -1, 1))))

# camera basis: +z toward the observer, +y the spin axis projected on screen (pole up)
_zc = _epf
_up = np.array([0.0, 0.0, 1.0]) - np.dot(_zc, [0, 0, 1.0]) * _zc
if np.linalg.norm(_up) < 1e-6:            # pole-on degenerate case
    _up = np.array([0.0, 1.0, 0.0]) - np.dot(_zc, [0, 1.0, 0]) * _zc
_yc = _up / np.linalg.norm(_up)
_xc = np.cross(_yc, _zc)
_Rcam = np.vstack([_xc, _yc, _zc])
_sun_cam = (_Rcam @ _spf).tolist()

def _mat2quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
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
    q = np.array(q); return (q / np.linalg.norm(q)).tolist()

CAMQ = _mat2quat(_Rcam)

# ---- physical properties, looked up from the production catalogue rather than hardcoded,
# so adding a target needs no edit here.
# IMPORTANT for the true-colour mode: TESS observes in ONE broad red bandpass, so these data
# carry NO colour information. The hue comes entirely from the published taxonomic class and
# geometric albedo; only the LUMINANCE variation across facets is derived from our photometry.
CATALOGUE = f'{Y34_CAT}'
try:
    _cat = pd.read_csv(CATALOGUE, low_memory=False)
    _row = _cat[_cat['designation'] == res['target']]
    _r = _row.iloc[0] if len(_row) else None
except Exception:
    _r = None

def _g(k, default=None):
    if _r is None or k not in _r.index:
        return default
    v = _r[k]
    return default if pd.isna(v) else v

GEO_ALBEDO = float(_g('albedo_real', 0.10))
SPEC = str(_g('spec_type', '')) or None
DIAM = _g('diameter_km_real')
HMAG = _g('magnitude_H')
LCDB_P = _g('published_rot_per_hr')
# Mild taxonomic colour slope: M-types are spectrally featureless with a modest red slope,
# C-types near-neutral and very dark. Illustrative tints, not spectrophotometry.
_TINT = {'M': [1.00, 0.93, 0.82], 'S': [1.00, 0.90, 0.76], 'C': [1.00, 0.98, 0.96]}
TINT = _TINT.get(SPEC, [1.0, 1.0, 1.0])


def _stat(label, value):
    return (f'<div class="stat-row"><span class="stat-label">{label}</span>'
            f'<span class="stat-value">{value}</span></div>')


PHYS = ''.join([
    _stat('Taxonomic type', SPEC if SPEC else '&mdash;'),
    _stat('Diameter', f'{DIAM:.1f} km' if DIAM is not None else '&mdash;'),
    _stat('Geometric albedo', f'{GEO_ALBEDO:.3f}'),
    _stat('Absolute mag <em>H</em>', f'{HMAG:.2f}' if HMAG is not None else '&mdash;'),
    _stat('LCDB period', f'{LCDB_P:.4f} hr' if LCDB_P is not None else 'none (new)'),
])

# cividis LUT (the project's standard heatmap colouring)
try:
    import matplotlib.cm as cm
    LUT = [[int(255 * c) for c in cm.get_cmap('cividis')(i / 31.0)[:3]] for i in range(32)]
except Exception:
    LUT = [[0, 32, 76], [60, 80, 110], [124, 123, 120], [200, 180, 100], [255, 233, 69]]

alb_note = ('' if not ALB else
            f'<div class="stat-row"><span class="stat-label">Albedo contrast</span>'
            f'<span class="stat-value">{(hi-lo)/np.mean(face_alb)*100:.0f}%</span></div>'
            f'<div class="stat-row"><span class="stat-label">Fit gain from albedo</span>'
            f'<span class="stat-value">{(1-ALB["rms_albedo"]/ALB["rms_uniform"])*100:.1f}%</span></div>')

DATA = json.dumps(dict(verts=verts.tolist(), facets=facets, alb=face_alb, weak=face_weak,
                       lo=lo, hi=hi, lut=LUT, lcP=lc_phase, lcF=lc_flux, lcE=lc_err, lcM=lc_model,
                       camQ=CAMQ, sunCam=_sun_cam, aspect=ASPECT,
                       geoAlbedo=GEO_ALBEDO, tint=TINT, spec=SPEC))

html = f'''<title>{res['target']} Shape Model</title>
<style>
  :root {{
    --bg:#090c12; --panel:#12161fcc; --panel-border:#232936; --hairline:#1c212c;
    --text:#e8ecf3; --text-dim:#8791a3; --text-faint:#545e70;
    --accent:#a78bfa; --accent-deep:#6d4bc7; --accent-bg:#2b1f52; --warn:#e0a458;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:"IBM Plex Sans",system-ui,sans-serif;height:100vh;overflow:hidden;position:relative}}
  canvas#c{{display:block;width:100%;height:100%;cursor:grab;touch-action:none}}
  canvas#c:active{{cursor:grabbing}}
  .panel{{position:absolute;background:var(--panel);border:1px solid var(--panel-border);
    border-radius:10px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}}
  .info{{top:20px;left:20px;max-width:360px;padding:18px 20px;overflow-y:auto;
    scrollbar-width:thin;scrollbar-color:var(--panel-border) transparent}}
  .info::-webkit-scrollbar{{width:7px}}
  .info::-webkit-scrollbar-thumb{{background:var(--panel-border);border-radius:4px}}
  .eyebrow{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
    letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint);margin:0 0 6px}}
  h1{{font-size:19px;font-weight:600;margin:0 0 12px;text-wrap:balance}}
  .stat-row{{display:flex;justify-content:space-between;align-items:baseline;padding:6px 0;
    border-top:1px solid var(--hairline);gap:16px}}
  .stat-row:first-of-type{{border-top:none}}
  .stat-label{{font-size:12.5px;color:var(--text-dim)}}
  .stat-value{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;
    font-variant-numeric:tabular-nums}}
  .sec{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;letter-spacing:.09em;
    text-transform:uppercase;color:var(--text-faint);margin:13px 0 3px}}
  .caveat{{margin-top:14px;padding:10px 12px;border-radius:7px;background:#e0a4581a;
    border:1px solid #e0a45840;font-size:11.5px;line-height:1.5;color:#e8c99a}}
  .ctrl{{bottom:20px;left:20px;padding:14px 16px;display:flex;flex-direction:column;gap:10px;
    min-width:300px}}
  .btn{{background:#1b2130;border:1px solid var(--panel-border);color:var(--text);
    border-radius:7px;padding:7px 12px;font-size:12.5px;cursor:pointer;font-family:inherit}}
  .btn:hover{{border-color:var(--accent)}}
  .btn[aria-pressed="true"]{{background:var(--accent-bg);border-color:var(--accent-deep);
    color:var(--accent)}}
  .btn:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
  .row{{display:flex;gap:8px;align-items:center}}
  label{{font-size:11.5px;color:var(--text-dim)}}
  input[type=range]{{flex:1;accent-color:var(--accent)}}
  canvas#lc{{width:100%;height:132px;display:block;border-radius:6px;background:#0d1119}}
  .hint{{margin:0;font-size:11px;line-height:1.45;color:var(--text-faint)}}
  .btn[disabled]{{opacity:.4;cursor:not-allowed}}
  .legend{{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text-dim);
    font-family:"IBM Plex Mono",ui-monospace,monospace}}
  .bar{{flex:1;height:9px;border-radius:5px}}
  @media (max-width:820px){{ .info{{max-width:260px;padding:12px 14px}} }}
</style>

<canvas id="c" role="img" aria-label="Interactive rotatable 3D convex shape model of asteroid {res['target']}, coloured by relative albedo"></canvas>

<div class="panel info">
  <p class="eyebrow">TESS convex inversion</p>
  <h1>{res['target']}</h1>
  <p class="sec" style="margin-top:0">Physical properties</p>
  {PHYS}
  <p class="sec">TESSELLATE solution</p>
  <div class="stat-row"><span class="stat-label">Rotation period</span><span class="stat-value">{res['adopted_period_hr']:.5f} hr</span></div>
  <div class="stat-row"><span class="stat-label">Amplitude</span><span class="stat-value">{AMP_MAG:.3f} mag</span></div>
  <div class="stat-row"><span class="stat-label">Equatorial ratio</span><span class="stat-value">{res.get('equatorial_ratio',float('nan')):.2f}</span></div>
  <div class="stat-row"><span class="stat-label">Polar / equatorial</span><span class="stat-value">{res.get('polar_ratio',float('nan')):.2f}</span></div>
  <div class="stat-row"><span class="stat-label">Observations</span><span class="stat-value">{res['n_observations']:,} pts / {res['n_sessions']} visits</span></div>
  <div class="stat-row"><span class="stat-label">Baseline</span><span class="stat-value">{res.get('baseline_days',0):.1f} d</span></div>
  <div class="stat-row"><span class="stat-label">Phase-angle range</span><span class="stat-value">{res['phase_angle_range_deg']:.2f} deg</span></div>
  <div class="stat-row"><span class="stat-label">Aspect angle (TESS)</span><span class="stat-value">{ASPECT:.1f} deg</span></div>
  <div class="stat-row"><span class="stat-label">Facets</span><span class="stat-value">{res['recovered_facets']}</span></div>
  <div class="stat-row"><span class="stat-label">Spin axis</span><span class="stat-value">assumed &lambda;={res.get('representative_lambda_deg',0):.0f}&deg;, &beta;={res.get('representative_beta_deg',0):+.0f}&deg;</span></div>
  {alb_note}
  <div class="caveat">
    Only {res['phase_angle_range_deg']:.2f}&deg; of phase-angle coverage &mdash; one viewing aspect.
    The spin axis is ASSUMED, not fitted: single-apparition data cannot determine a pole, and
    every orientation fits this lightcurve about equally well. The shape is barely affected by
    that choice, but its orientation in space carries no information.
    Albedo colouring assumes all residual is albedo rather than unmodelled shape; the two are
    degenerate in disc-integrated photometry.
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
  <canvas id="lc" aria-label="Phase-folded lightcurve, one period, with a marker tracking the displayed rotation"></canvas>
</div>

<script>
const D = {DATA};
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const lcv = document.getElementById('lc'), lctx = lcv.getContext('2d');
let dpr = Math.min(window.devicePixelRatio||1, 2);
function size(){{
  for (const [el,g] of [[cv,ctx],[lcv,lctx]]){{
    const r = el.getBoundingClientRect();
    el.width = Math.max(1,r.width*dpr); el.height = Math.max(1,r.height*dpr);
    g.setTransform(dpr,0,0,dpr,0,0);
  }}
}}
window.addEventListener('resize', size); size();

// Keep the info panel clear of the control panel. Both are absolutely positioned in the same
// corner column, so as the info panel grew it began overlapping the lightcurve; cap it against
// the control panel's MEASURED height rather than a guessed constant, and re-measure on resize.
function fitPanels(){{
  const info=document.querySelector('.info'), ctrl=document.querySelector('.ctrl');
  if(!info||!ctrl) return;
  const avail = window.innerHeight - ctrl.getBoundingClientRect().height - 20 - 20 - 16;
  info.style.maxHeight = Math.max(160, avail) + 'px';
}}
window.addEventListener('resize', fitPanels);

// ---- colour ramps. Two modes:
//   albedo -- cividis over the RELATIVE albedo map, for reading structure.
//   true   -- what the eye would see: relative albedo x the published geometric albedo gives
//             a real reflectance, sRGB-encoded so the darkness is perceptually right, then a
//             mild taxonomic tint. At p=0.117 this is genuinely dark, like charcoal; a body
//             rendered bright grey would badly misrepresent an asteroid surface.
let colourMode='albedo';
function srgb(x){{ x=Math.max(0,Math.min(1,x));
  return x<=0.0031308 ? 12.92*x : 1.055*Math.pow(x,1/2.4)-0.055; }}
function trueColour(a){{
  const refl = D.geoAlbedo*a;
  const e = srgb(refl);
  return [e*255*D.tint[0], e*255*D.tint[1], e*255*D.tint[2]];
}}
function ramp(a){{
  if(colourMode==='true') return trueColour(a);
  const t = D.hi>D.lo ? (a-D.lo)/(D.hi-D.lo) : 0.5;
  const i = Math.max(0, Math.min(D.lut.length-1, Math.round(t*(D.lut.length-1))));
  return D.lut[i];
}}


// ---- quaternions for free drag
function qMul(a,b){{return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
  a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
  a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];}}
function qAxis(ax,an){{const s=Math.sin(an/2);return [Math.cos(an/2),ax[0]*s,ax[1]*s,ax[2]*s];}}
function qNorm(q){{const n=Math.hypot(...q);return q.map(v=>v/n);}}
function qMat(q){{const [w,x,y,z]=q;return [
  [1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
  [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
  [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];}}

let orient = qAxis([1,0,0],-1.2), zoom=1, phase=0, spinning=false;
const scale = (()=>{{let m=0;for(const v of D.verts)m=Math.max(m,Math.hypot(v[0],v[1],v[2]));return m||1;}})();

function draw(){{
  const w=cv.width/dpr, h=cv.height/dpr;
  ctx.clearRect(0,0,w,h);
  const R = qMat(orient);
  const a = 2*Math.PI*phase, ca=Math.cos(a), sa=Math.sin(a);
  const S = Math.min(w,h)*0.33*zoom/scale;
  // Centre the body in the space ABOVE the control panel rather than in the whole canvas,
  // so it never clips the lightcurve. Measured from the panel each frame so it stays correct
  // when the window resizes or the panel reflows.
  const panelH = (document.querySelector('.ctrl')?.getBoundingClientRect().height || 0) + 28;
  const cy = Math.max(S*0.9, (h - panelH)/2);
  // Body spin about z, the spin axis by convexinv's convention. Sign VERIFIED against the
  // data, not reasoned from the transform: rendering the mesh at each phase and correlating
  // the synthetic brightness with the observed folded curve gives r=+0.99 this way and
  // r=-0.69 reversed (validate_viewer_sync.py). Do not "fix" it without re-running that.
  const P = D.verts.map(v=>{{
    const x=v[0]*ca - v[1]*sa, y=v[0]*sa + v[1]*ca, z=v[2];
    return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z,
            R[2][0]*x+R[2][1]*y+R[2][2]*z];
  }});
  const faces = D.facets.map((f,i)=>{{
    const p=f.map(k=>P[k]);
    // NEWELL'S METHOD, summed over every edge. Faces here are polygons of 4-9 vertices, and
    // taking the normal from just the first three is badly conditioned: 180 of Eurydike's 289
    // faces have a near-collinear leading triplet (face 258: cross product 2.6e-5 against a
    // true area of 0.12). The resulting direction is numerically arbitrary and jitters between
    // frames, flipping both the shading and the backface test -- that was the flickering facet.
    let n=[0,0,0];
    for(let k=0;k<p.length;k++){{
      const a=p[k], b=p[(k+1)%p.length];
      n[0]+=(a[1]-b[1])*(a[2]+b[2]);
      n[1]+=(a[2]-b[2])*(a[0]+b[0]);
      n[2]+=(a[0]-b[0])*(a[1]+b[1]);
    }}
    const nl=Math.hypot(...n)||1; n=n.map(c=>c/nl);
    const cz=p.reduce((s,q)=>s+q[2],0)/p.length;
    const cen=[p.reduce((s,q)=>s+q[0],0)/p.length,p.reduce((s,q)=>s+q[1],0)/p.length,cz];
    if(n[0]*cen[0]+n[1]*cen[1]+n[2]*cen[2] < 0) n=n.map(c=>-c);
    return {{p,n,cz,i}};
  }}).filter(f=>f.n[2]>0).sort((A,B)=>A.cz-B.cz);
  for(const f of faces){{

    // In locked mode light the body from the real Sun direction, so the terminator is where
    // TESS saw it; in free mode there is no meaningful illumination geometry, so shade from
    // the camera instead.
    const lit = (mode==='locked')
      ? 0.18 + 0.82*Math.max(0, f.n[0]*D.sunCam[0]+f.n[1]*D.sunCam[1]+f.n[2]*D.sunCam[2])
      : 0.25 + 0.75*Math.max(0, f.n[2]);
    const c = ramp(D.alb[f.i]);
    ctx.beginPath();
    f.p.forEach((q,k)=>{{const X=w/2+q[0]*S, Y=cy-q[1]*S; k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}});
    ctx.closePath();
    ctx.fillStyle = `rgb(${{Math.round(c[0]*lit)}},${{Math.round(c[1]*lit)}},${{Math.round(c[2]*lit)}})`;
    ctx.fill();
    ctx.strokeStyle = D.weak[f.i] ? 'rgba(224,164,88,0.55)' : 'rgba(255,255,255,0.10)';
    ctx.lineWidth = D.weak[f.i] ? 1.1 : 0.5;
    ctx.stroke();
  }}
  drawLC();
}}

function drawLC(){{
  // Axes with real ticks and labels: the panel is a measurement, not a sparkline, and without
  // a flux scale the amplitude is unreadable.
  const w=lcv.width/dpr, h=lcv.height/dpr;
  const L=58, Rp=8, T=8, B=26;    // L leaves room for tick text AND the rotated y label
  lctx.clearRect(0,0,w,h);
  if(!D.lcP.length) return;
  const eArr=D.lcE||D.lcF.map(()=>0);
  let vals=D.lcF.map((f,i)=>f-eArr[i]).concat(D.lcF.map((f,i)=>f+eArr[i]));
  if(D.lcM) vals=vals.concat(D.lcM);
  const fmin=Math.min(...vals), fmax=Math.max(...vals);
  const rng=(fmax-fmin)||1, lo=fmin-0.12*rng, hi=fmax+0.12*rng;
  const X=p=>L+p*(w-L-Rp), Y=f=>h-B-((f-lo)/(hi-lo))*(h-T-B);

  lctx.strokeStyle='#2b3342'; lctx.lineWidth=1;
  lctx.beginPath(); lctx.moveTo(L,T); lctx.lineTo(L,h-B); lctx.lineTo(w-Rp,h-B); lctx.stroke();
  lctx.fillStyle='#8791a3'; lctx.font='9px ui-monospace,monospace';

  lctx.textAlign='center'; lctx.textBaseline='top';
  for(const t of [0,0.25,0.5,0.75,1]){{
    const x=X(t);
    lctx.strokeStyle='#2b3342'; lctx.beginPath();
    lctx.moveTo(x,h-B); lctx.lineTo(x,h-B+4); lctx.stroke();
    lctx.fillText(t.toFixed(2),x,h-B+6);
  }}
  lctx.textAlign='right'; lctx.textBaseline='middle';
  for(const fv of [lo+(hi-lo)*0.12, (lo+hi)/2, hi-(hi-lo)*0.12]){{
    const y=Y(fv);
    lctx.strokeStyle='#2b3342'; lctx.beginPath();
    lctx.moveTo(L-4,y); lctx.lineTo(L,y); lctx.stroke();
    lctx.fillText(fv.toFixed(3),L-6,y);
  }}
  lctx.fillStyle='#545e70';
  lctx.textAlign='center'; lctx.textBaseline='bottom';
  lctx.fillText('Rotational phase',(L+w-Rp)/2,h-1);
  lctx.save(); lctx.translate(11,(T+h-B)/2); lctx.rotate(-Math.PI/2);
  lctx.textAlign='center'; lctx.textBaseline='top';
  lctx.fillText('Relative flux',0,0); lctx.restore();

  // model first, so the data sits on top of it
  if(D.lcM){{
    lctx.save(); lctx.setLineDash([5,3]);
    lctx.strokeStyle='#e06c75'; lctx.lineWidth=1.5; lctx.beginPath();
    D.lcP.forEach((p,i)=>{{i?lctx.lineTo(X(p),Y(D.lcM[i])):lctx.moveTo(X(p),Y(D.lcM[i]));}});
    lctx.stroke(); lctx.restore();
  }}
  // observed: error bar then marker
  lctx.strokeStyle='#7c6bb0'; lctx.lineWidth=1;
  D.lcP.forEach((p,i)=>{{
    const e=eArr[i]; if(!(e>0)) return;
    const x=X(p);
    lctx.beginPath(); lctx.moveTo(x,Y(D.lcF[i]-e)); lctx.lineTo(x,Y(D.lcF[i]+e)); lctx.stroke();
  }});
  lctx.fillStyle='#a78bfa';
  D.lcP.forEach((p,i)=>{{
    lctx.beginPath(); lctx.arc(X(p),Y(D.lcF[i]),1.9,0,2*Math.PI); lctx.fill();
  }});
  if(D.lcM){{
    lctx.font='9px ui-monospace,monospace'; lctx.textAlign='right'; lctx.textBaseline='top';
    lctx.fillStyle='#e06c75'; lctx.fillText('- - convex model', w-Rp, T);
  }}

  const x=X(phase);
  lctx.strokeStyle='#e0a458'; lctx.lineWidth=1.4;
  lctx.beginPath(); lctx.moveTo(x,T); lctx.lineTo(x,h-B); lctx.stroke();
  let j=0; while(j<D.lcP.length-1 && D.lcP[j+1]<phase) j++;
  const f0=D.lcF[j], f1=D.lcF[Math.min(j+1,D.lcF.length-1)];
  const p0=D.lcP[j], p1=D.lcP[Math.min(j+1,D.lcP.length-1)];
  const t=(p1>p0)?(phase-p0)/(p1-p0):0;
  lctx.fillStyle='#e0a458'; lctx.beginPath();
  lctx.arc(x, Y(f0+(f1-f0)*t), 3.2, 0, 2*Math.PI); lctx.fill();
}}

// ---- interaction. Two modes:
//   free   -- drag reorients the VIEW (quaternion); phase is unchanged, so the lightcurve
//             marker stays put. Use it to inspect the body from any direction.
//   locked -- drag rotates the BODY about its spin axis and nothing else, so horizontal drag
//             IS the rotational phase and the lightcurve marker tracks it exactly. Play/pause
//             belongs to this mode only: animating a free-dragged view would not correspond
//             to any physical rotation.
let mode='free';
const slider=document.getElementById('ph'), phv=document.getElementById('phv');
const spinBtn=document.getElementById('spin'), hint=document.getElementById('hint');
const mFree=document.getElementById('mFree'), mLock=document.getElementById('mLock');
function setPhase(p){{ phase=((p%1)+1)%1; slider.value=Math.round(phase*1000); phv.textContent=phase.toFixed(3); }}
function setMode(m){{
  mode=m;
  const lock = m==='locked';
  mFree.setAttribute('aria-pressed', lock?'false':'true');
  mLock.setAttribute('aria-pressed', lock?'true':'false');
  spinBtn.disabled = !lock;
  if(!lock && spinning){{ spinning=false; spinBtn.setAttribute('aria-pressed','false');
    spinBtn.innerHTML='&#9654; Play'; }}
  hint.innerHTML = lock
    ? 'Snapped to the TESS viewing geometry. Drag horizontally to turn the body about its spin axis &mdash; the lightcurve marker follows. Press Play to rotate at a steady rate.'
    : 'Drag to rotate the view freely. Switch to <em>Locked to axis</em> to drag the body around its spin axis, synced to the lightcurve.';
  cv.style.cursor = lock ? 'ew-resize' : 'grab';
  if(lock) orient = D.camQ.slice();   // snap to the TESS viewing geometry
  draw();
}}
// Surface-colouring toggle removed: it gave the mesh and the info table more room, and
// with the per-facet albedo map disabled the two modes differed only by a flat tint.
// setColour is kept as a no-op so the existing init call needs no special-casing.
function setColour(){{ colourMode='albedo'; draw(); }}
mFree.addEventListener('click', ()=>setMode('free'));
mLock.addEventListener('click', ()=>setMode('locked'));
slider.addEventListener('input', ()=>{{ setPhase(slider.value/1000); if(!spinning) draw(); }});
spinBtn.addEventListener('click', ()=>{{
  if(mode!=='locked') return;
  spinning=!spinning;
  spinBtn.setAttribute('aria-pressed', spinning?'true':'false');
  spinBtn.innerHTML = spinning ? '&#10073;&#10073; Pause' : '&#9654; Play';
}});
document.getElementById('reset').addEventListener('click', ()=>{{
  orient=qAxis([1,0,0],-1.2); zoom=1; setPhase(0); draw();
}});
let drag=false,lx=0,ly=0;
cv.addEventListener('pointerdown',e=>{{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);}});
window.addEventListener('pointermove',e=>{{
  if(!drag) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(mode==='locked'){{
    setPhase(phase + dx/360);          // a full drag across ~360 px is one rotation
    draw();
  }} else {{
    orient=qNorm(qMul(qAxis([1,0,0],dy*0.008), qMul(qAxis([0,1,0],dx*0.008), orient)));
    if(!spinning) draw();
  }}
}});
window.addEventListener('pointerup',()=>{{drag=false;}});
cv.addEventListener('wheel',e=>{{e.preventDefault();zoom*=e.deltaY<0?1.08:0.93;
  zoom=Math.max(0.4,Math.min(zoom,3.5)); if(!spinning) draw();}},{{passive:false}});

const PERIOD_MS = 8000;
let last=performance.now();
function tick(now){{
  const dt=now-last; last=now;
  if(spinning){{ setPhase(phase + dt/PERIOD_MS); draw(); }}
  requestAnimationFrame(tick);
}}
fitPanels(); setPhase(0); setColour('albedo'); setMode('locked');
// autoplay: the body is rotating as soon as the page opens, which is what makes the shape and
// its lightcurve legible at a glance. setMode('locked') must run FIRST -- it clears `spinning`
// when leaving locked mode, so setting the flag before it would be undone immediately.
spinning = true;
spinBtn.setAttribute('aria-pressed', 'true');
spinBtn.innerHTML = '&#10073;&#10073; Pause';
requestAnimationFrame(tick);
window.addEventListener('resize', draw);
</script>
'''

out = os.path.join(HERE, f'{TAG}_shape_viewer.html')
open(out, 'w').write(html)
print(f'wrote {out}  ({len(html)/1024:.0f} KB, {len(facets)} facets, '
      f'albedo {"on" if ALB else "off"}, {len(lc_phase)} lightcurve bins)')
