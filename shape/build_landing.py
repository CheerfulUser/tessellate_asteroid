"""Build the public landing page: headline figures, live search, and the catalog index.

SEARCH DESIGN. 16,427 objects is too many to ship as one blob and too few to justify a backend.
The whole index is ~600 KB of JSON (designation, period, amplitude, size, flags) which is a
single cached fetch, and filtering happens client-side. That keeps the site static -- no server,
nothing to break -- while still giving instant search over the full catalog.

Only objects that actually have a page are linked. The rest of the catalog is searchable and
its numbers are shown, but a link is only rendered where a shape model exists, so nobody follows
a dead URL.
"""
import hashlib, json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')
CAT = f'{Y34}/population_figs/all_sector_report_v5_doubled.csv'
FIGS = f'{Y34}/population_figs/web'


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


# Taxonomic COMPLEX, not the raw code. Only 3.7% of objects have a type at all, and the values
# mix Tholen and Bus-DeMeo with qualifiers -- Ch, Xk, BCU, "CX:" -- so filtering on the literal
# strings would give dozens of near-empty categories. Grouping into the standard complexes
# (C includes B/F/G, S includes A/K/L/Q/R, X includes E/M/P) gives categories a user can
# actually pick from.
_COMPLEX = {'B': 'C', 'C': 'C', 'F': 'C', 'G': 'C',
            'A': 'S', 'K': 'S', 'L': 'S', 'Q': 'S', 'R': 'S', 'S': 'S',
            'E': 'X', 'M': 'X', 'P': 'X', 'X': 'X',
            'D': 'D', 'T': 'D', 'V': 'V'}


def complex_of(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    c = str(t).strip()
    return _COMPLEX.get(c[:1].upper()) if c else None


c = pd.read_csv(CAT, low_memory=False)
r = c[(c.is_reliable == True) & c.period_hr.notna()].copy()   # noqa: E712
try:
    amp = pd.read_csv(f'{Y34}/comparison_data/cluster_features.csv')[['designation', 'amplitude_mag']]
    r = r.merge(amp, on='designation', how='left')
except FileNotFoundError:
    r['amplitude_mag'] = np.nan
try:
    sc = pd.read_csv(f'{Y34}/comparison_data/profile_scores.csv')[
        ['designation', 'asymmetry', 'outlier_rank']]
    r = r.merge(sc, on='designation', how='left')
except FileNotFoundError:
    r['asymmetry'] = np.nan; r['outlier_rank'] = np.nan

# Lightcurve symmetry, from the ratio of odd- to even-harmonic power in the Fourier fit.
# A rotating triaxial body gives two near-identical maxima per cycle -- pure even harmonics;
# odd power is what makes the two halves differ.
#
# The monomodal cut is measured, not chosen: objects the pipeline independently flagged
# double_peaked=False sit at median odd/even 2.57 against 0.35 for the rest, and 1.32 separates
# the two populations at 99.3%. The symmetric/asymmetric cut at 0.25 is the lower quartile of
# the double-peaked population, so "asymmetric" means visibly uneven maxima rather than a
# physical threshold. Independent check: the most elongated bodies (amplitude > 0.5 mag) are
# also the most symmetric, median 0.19, as a triaxial ellipsoid should be.
MONOMODAL_OE = 1.32       # signed scale -0.138
SYMMETRIC_OE = 0.25       # signed scale +0.600
try:
    ho = pd.read_csv(f'{Y34}/comparison_data/cluster_features.csv')[
        ['designation', 'odd_over_even']]
    r = r.merge(ho, on='designation', how='left')
    # cluster_features.csv holds the harmonics of the fold as it was when that file was
    # built. For every object whose period was later corrected it therefore describes the WRONG
    # fold, and the error is not subtle: at twice the true period the fold contains two
    # near-identical copies of one rotation, odd power collapses, and an asymmetric object is
    # published as symmetric. (453) Tea read +0.96 measured at 13.6192 h against its adopted
    # 6.8096 h. symmetry_fixed.csv carries odd/even re-measured at the adopted period for every
    # affected object; it supersedes the earlier doubled-only override.
    fix = pd.read_csv(f'{ROOT}/data/symmetry_fixed.csv')[['designation', 'odd_over_even']]
    fix = fix.rename(columns={'odd_over_even': '_oe_fix'})
    r = r.merge(fix, on='designation', how='left')
    n_fix = int(r._oe_fix.notna().sum())
    r['odd_over_even'] = r._oe_fix.where(r._oe_fix.notna(), r.odd_over_even)
    r = r.drop(columns=['_oe_fix'])
    print(f'  symmetry re-measured at the adopted period for {n_fix} objects')
    oe = r.odd_over_even.replace([np.inf, -np.inf], np.nan)
    r['sym'] = np.where(oe.isna(), None,
               np.where(oe >= MONOMODAL_OE, 'monomodal',
               np.where(oe < SYMMETRIC_OE, 'symmetric', 'asymmetric')))
except FileNotFoundError:
    r['odd_over_even'] = np.nan; r['sym'] = None
print('  symmetry: ' + ', '.join(f'{k}={v:,}' for k, v in r.sym.value_counts().items()))

have_page = {f[:-5] for f in os.listdir(f'{ROOT}/asteroid')} if os.path.isdir(f'{ROOT}/asteroid') else set()
_shape_keys = ({f[:-5] for f in os.listdir(f'{ROOT}/data/shapes')}
                if os.path.isdir(f'{ROOT}/data/shapes') else set())
r['key'] = r.designation.map(sn)
r['has_page'] = r.key.isin(have_page)
print(f'{len(r):,} reliable objects, {int(r.has_page.sum()):,} with a published page')

os.makedirs(f'{ROOT}/assets', exist_ok=True)
for f in ('spin_size.png', 'period_hist.png', 'amplitude_period.png', 'orbital_elements.png'):
    src = f'{FIGS}/{f}'
    if os.path.exists(src):
        open(f'{ROOT}/assets/{f}', 'wb').write(open(src, 'rb').read())

# compact search index: short keys, rounded values -- this is fetched by every visitor
idx = [dict(d=row.designation, k=row.key, p=round(float(row.period_hr), 4),
            a=None if pd.isna(row.amplitude_mag) else round(float(row.amplitude_mag), 3),
            s=None if pd.isna(row.diameter_km_real) else round(float(row.diameter_km_real), 1),
            t=None if pd.isna(row.spec_type) else str(row.spec_type),
            c=complex_of(None if pd.isna(row.spec_type) else row.spec_type),
            l=None if pd.isna(row.published_rot_per_hr) else round(float(row.published_rot_per_hr), 4),
            y=row.sym,
            o=None if pd.isna(row.odd_over_even) else round(float(row.odd_over_even), 3),
            m=None if pd.isna(row.odd_over_even) else round((1.0-float(row.odd_over_even))/(1.0+float(row.odd_over_even)), 3),
            g=bool(row.key in _shape_keys))      # has a SHAPE MODEL; every object has a page
       for row in r.itertuples()]
json.dump(idx, open(f'{ROOT}/data/index.json', 'w'), separators=(',', ':'))
# Cache-bust the data URL with a content hash. The pages fetch index.json from a fixed path, so
# a returning visitor's browser happily serves the previous copy against the new HTML -- which
# renders as a column of em-dashes for any field the old file predates. The hash changes only
# when the data does, so unchanged builds still hit cache.
IDXV = hashlib.sha1(open(f'{ROOT}/data/index.json', 'rb').read()).hexdigest()[:8]
print(f'  index.json {os.path.getsize(f"{ROOT}/data/index.json")/1024:.0f} KB')

S = dict(n=len(r), new=int(r.published_rot_per_hr.isna().sum()),
         lcdb=int(r.published_rot_per_hr.notna().sum()),
         med=r.period_hr.median(), long=int((r.period_hr > 24).sum()),
         base=float((r.baseline_hr / 24).median()),
         rots=int((r.baseline_hr / r.period_hr).median()),
         pts=int(r.n_points.median()),
         pages=int(r.has_page.sum()),
         shapes=int(r.key.isin(_shape_keys).sum()),
         noshape=int((~r.key.isin(_shape_keys)).sum()))

import landing_text as TXT   # noqa: E402  (all page prose lives there)


_BARE_AMP = re.compile(r'&(?!(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});)')


def _t(v):
    """Fill {placeholders} from the catalogue stats and collapse the source line wrapping.

    Also escapes bare ampersands. The copy is hand-edited plain text, so "rotation & shape" is
    natural to write but is invalid HTML and would be swallowed as an entity if a word followed
    it. Entities that are already written out, like &mdash;, are left alone."""
    return _BARE_AMP.sub('&amp;', ' '.join(str(v).format(**S).split()))


def _check_markup(name, v):
    """Fail the build on unbalanced inline tags in hand-edited copy.

    An unclosed <a> does not look broken in the source but swallows the rest of the paragraph
    into the link, and a stray '">' from an interrupted edit renders as literal text. Both
    shipped once. Cheap to catch here, invisible until someone reads the page otherwise."""
    for tag in ('a', 'strong', 'em'):
        o = len(re.findall(rf'<{tag}\b', v))
        c = len(re.findall(rf'</{tag}>', v))
        if o != c:
            raise SystemExit(f'landing_text.{name}: {o} <{tag}> but {c} </{tag}> -- '
                             f'unbalanced tag, fix before building')
    if re.search(r'">[^<]*">', v):
        raise SystemExit(f'landing_text.{name}: stray \'">\' inside an attribute -- '
                         f'looks like a half-finished edit')


T = {}
for k, v in vars(TXT).items():
    if k.isupper() and isinstance(v, str):
        _check_markup(k, v)
        T[k] = _t(v)
T['KPIS'] = ''.join(
    f'<div><div class="v">{_t(val)}</div><div class="l">{_t(lab)}</div></div>'
    for val, lab in TXT.KPIS)
for _k, _v in TXT.CAPTIONS.items():
    _check_markup(f'CAPTIONS[{_k}]', _v)
T['CAPTIONS'] = {k: _t(v) for k, v in TXT.CAPTIONS.items()}

html = f'''<!doctype html><meta charset="utf-8">
<title>TESSELLATE asteroid rotation catalog</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="assets/viewer.css">
<style>
  body{{overflow:visible;height:auto;position:static}}
  main{{max-width:940px;margin:0 auto;padding:44px 24px 72px}}
  h1{{font-size:30px;margin:0 0 8px;letter-spacing:-0.02em}}
  .lede{{color:var(--text-dim);font-size:14.5px;line-height:1.65;margin:0 0 26px;max-width:70ch}}
  .kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 30px}}
  /* direct child only: ".kpi div" also matched the .v and .l inside each tile, so every
     number and label got its own border inside the card */
  .kpi > div{{background:var(--panel);border:1px solid var(--panel-border);border-radius:9px;
    padding:14px 16px}}
  .kpi .v{{font-family:"IBM Plex Mono",monospace;font-size:21px;font-variant-numeric:tabular-nums}}
  .kpi .l{{font-size:11.5px;color:var(--text-dim);margin-top:3px;line-height:1.35}}
  h2{{font-size:16px;margin:36px 0 10px}}
  p.note{{color:var(--text-dim);font-size:13px;line-height:1.6;margin:0 0 14px;max-width:72ch}}
  .figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
  /* citation links inside body prose */
  .note a{{color:var(--accent);text-decoration:none;border-bottom:1px solid #a78bfa55}}
  .note a:hover{{border-bottom-color:var(--accent)}}
  .note a:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
  /* opaque, and the same literal _figstyle renders the figures on -- a translucent card
     composites to a colour the figure cannot know, so they drifted apart */
  .figs figure{{margin:0;background:#0f141c;border:1px solid var(--panel-border);
    border-radius:9px;padding:10px}}
  .figs img{{width:100%;height:auto;border-radius:5px;display:block}}
  .figs figcaption{{font-size:11.5px;color:var(--text-dim);margin-top:7px;line-height:1.45}}
  .searchrow{{display:flex;gap:10px;align-items:stretch}}
  .searchrow #q{{flex:1;min-width:0}}
  #rand{{flex:0 0 auto;padding:0 16px;font-family:inherit;font-size:13px;cursor:pointer;
    color:var(--accent);background:#141a26;border:1px solid var(--panel-border);
    border-radius:8px;white-space:nowrap}}
  #rand:hover{{border-color:var(--accent);background:#18202e}}
  #rand:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
  @media (max-width:520px){{ .searchrow{{flex-wrap:wrap}} #rand{{width:100%;padding:10px 16px}} }}
  /* ---- MOBILE. Desktop rules above are untouched. The table is the problem: seven columns
     of numbers will not fit a phone, so it scrolls inside its own container rather than
     forcing the whole page sideways, and the least essential columns drop out. */
  @media (max-width:640px){{
    main{{padding:26px 14px 56px}}
    h1{{font-size:26px}}
    .kpi{{grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}}
    .figs{{grid-template-columns:1fr}}
    .tablewrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
    table{{font-size:12px;min-width:460px}}
    th,td{{padding:6px 7px}}
    /* Diameter and LCDB are the least load-bearing on a small screen */
    th:nth-child(4),td:nth-child(4),th:nth-child(7),td:nth-child(7){{display:none}}
  }}

  #q{{width:100%;padding:11px 13px;font-size:14px;border-radius:8px;background:#141a26;
    border:1px solid var(--panel-border);color:var(--text);font-family:inherit}}
  #q:focus{{outline:none;border-color:var(--accent)}}
  .filters{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 4px;font-size:12px;color:var(--text-dim)}}
  .filters label{{display:flex;align-items:center;gap:5px;cursor:pointer}}
  #hits{{font-size:12px;color:var(--text-faint);margin:8px 0}}
  table{{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;
    font-variant-numeric:tabular-nums}}
  th{{text-align:right;padding:7px 9px;border-bottom:1px solid var(--panel-border);
    color:var(--text-faint);font-weight:500;font-size:11.5px;letter-spacing:.04em}}
  /* sortable headers: the arrow is reserved space at all times so the row does not
     reflow when sorting moves from one column to another */
  th[data-k]{{cursor:pointer;user-select:none;white-space:nowrap}}
  th[data-k]:hover,th[data-k]:focus-visible{{color:var(--text)}}
  th[data-k]:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px}}
  th[data-k]::after{{content:'\\2195';margin-left:5px;opacity:.25;font-size:10px}}
  th[aria-sort="ascending"]::after{{content:'\\2191';opacity:1;color:var(--accent)}}
  th[aria-sort="descending"]::after{{content:'\\2193';opacity:1;color:var(--accent)}}
  th[aria-sort]{{color:var(--text)}}
  td{{text-align:right;padding:7px 9px;border-bottom:1px solid var(--hairline)}}
  th:first-child,td:first-child{{text-align:left}}
  tbody tr:hover{{background:#161d2b}}
  td a{{color:var(--accent);text-decoration:none}}
  /* symmetry tags. No yellow; each tint reads on the dark ground and the label carries the
     meaning, so the colour is reinforcement rather than the only encoding. */
  .tag{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
    letter-spacing:.02em;white-space:nowrap;border:1px solid transparent}}
  .tag.symmetric{{color:#5ec8b4;background:rgba(94,200,180,.11);border-color:rgba(94,200,180,.28)}}
  .tag.asymmetric{{color:#e09a5a;background:rgba(224,154,90,.11);border-color:rgba(224,154,90,.28)}}
  .tag.monomodal{{color:#b294e0;background:rgba(178,148,224,.12);border-color:rgba(178,148,224,.30)}}
  td a:hover{{text-decoration:underline}}
  /* Every object links now. Marking the no-model rows by DIMMING them was a mistake: dim grey
     was previously the not-a-link colour, so a working link looked dead. Keep the link colour
     and carry the distinction in an explicit badge instead. */
  .lcflag{{display:inline-block;margin-left:6px;padding:0 5px;border-radius:4px;font-size:9.5px;
    letter-spacing:.04em;color:var(--text-faint);border:1px solid var(--panel-border);
    vertical-align:1px}}
  a.more{{color:var(--accent);text-decoration:none;font-weight:600}}
  a.more:hover{{text-decoration:underline}}
  .caveat{{margin-top:30px;padding:13px 15px;border-radius:8px;background:#e0a4581a;
    border:1px solid #e0a45840;font-size:12.5px;line-height:1.6;color:#e8c99a;max-width:72ch}}
</style>
<main>
  <p class="eyebrow">{T['EYEBROW']}</p>
  <h1>{T['TITLE']}</h1>
  <p class="lede">{T['LEDE']}</p>

  <div class="kpi">{T['KPIS']}</div>

  <h2>{T['SEARCH_HEADING']}</h2>
  <p class="note">{T['SEARCH_NOTE']}</p>
  <div class="searchrow">
    <input id="q" type="search" placeholder="{T['SEARCH_PLACEHOLDER']}" autocomplete="off">
    <button id="rand" type="button" title="Open a random asteroid from the catalog">{T['RANDOM_BUTTON']}</button>
  </div>
  <div id="hits"></div>
  <div class="tablewrap"><table><thead><tr>
    <th data-k="d" tabindex="0" role="button">Object</th>
    <th data-k="p" tabindex="0" role="button">Period (hours)</th>
    <th data-k="a" tabindex="0" role="button">Amplitude</th>
    <th data-k="s" tabindex="0" role="button">Diameter (km)</th>
    <th data-k="t" tabindex="0" role="button">Type</th>
    <th data-k="m" tabindex="0" role="button" title="+1 symmetric (two identical maxima), 0 equal odd and even power, -1 monomodal (one maximum per rotation)">Symmetry</th>
    <th data-k="l" tabindex="0" role="button">LCDB (hours)</th>
  </tr></thead><tbody id="rows"></tbody></table></div>
  <p class="note" style="margin-top:12px"><a class="more" href="search.html">{T['BROWSE_LINK']}</a> &mdash; {T['BROWSE_NOTE']}</p>

  <h2>{T['POPULATION_HEADING']}</h2>
  <div class="figs">
    <figure><img src="assets/period_hist.png" alt="Distribution of rotation periods">
      <figcaption>{T['CAPTIONS']['period_hist.png']}</figcaption></figure>
    <figure><img src="assets/spin_size.png" alt="Spin rate against diameter">
      <figcaption>{T['CAPTIONS']['spin_size.png']}</figcaption></figure>
    <figure><img src="assets/amplitude_period.png" alt="Lightcurve amplitude against period">
      <figcaption>{T['CAPTIONS']['amplitude_period.png']}</figcaption></figure>
    <figure style="grid-column:1/-1"><img src="assets/orbital_elements.png"
        alt="Semi-major axis against inclination and eccentricity">
      <figcaption>{T['CAPTIONS']['orbital_elements.png']}</figcaption></figure>
  </div>

  <h2>{T['DOWNLOAD_HEADING']}</h2>
  <p class="note">{T['DOWNLOAD_NOTE']}</p>

  <h2>{T['DATA_HEADING']}</h2>
  <p class="note">{T['DATA']}</p>

  <h2>{T['CREDIT_HEADING']}</h2>
  <p class="note">{T['CREDIT']}</p>

  <div class="caveat">
    <strong>{T['CAVEAT_TITLE']}</strong> {T['CAVEAT']}
  </div>
</main>
<script>
const IDX_URL='data/index.json?v={IDXV}';
const rows=document.getElementById('rows'), hits=document.getElementById('hits'),
      q=document.getElementById('q');
let DATA=[], sortKey=null, sortDir=1;
const fmt=(v,d)=>v==null?'&mdash;':(+v).toFixed(d);
const SYMLBL={{symmetric:'Symmetric',asymmetric:'Asymmetric',monomodal:'Monomodal'}};
function sym(o){{
  if(o.m==null) return '&mdash;';
  const t=` title="${{SYMLBL[o.y]||''}} — signed symmetry ${{o.m}} (odd/even ratio ${{o.o}})"`;
  return `<span class="tag ${{o.y}}"${{t}}>${{o.m>0?'+':''}}${{o.m.toFixed(2)}}</span>`;
}}
const PREVIEW=10;
const LABEL={{d:'name',p:'period',a:'amplitude',s:'diameter',t:'type',m:'symmetry',l:'LCDB period'}};
function render(list){{
  let note='';
  if(list.length>PREVIEW){{
    if(!sortKey) note=` — showing ${{PREVIEW}}`;
    else if(sortKey==='m')                      // high index = most symmetric
      note=` — showing the ${{PREVIEW}} ${{sortDir>0?'least':'most'}} symmetric`;
    else note=` — showing the ${{PREVIEW}} ${{sortDir>0?'lowest':'highest'}} by `+LABEL[sortKey];
  }}
  hits.textContent=`${{list.length.toLocaleString()}} of ${{DATA.length.toLocaleString()}} objects`+note;
  rows.innerHTML=list.slice(0,PREVIEW).map(o=>{{
    const name=`<a href="asteroid/${{o.k}}.html">${{o.d}}</a>`+(o.g?'':'<span class="lcflag" title="Lightcurve only &mdash; no shape model for this object">lc</span>');
    return `<tr><td>${{name}}</td><td>${{fmt(o.p,4)}}</td><td>${{fmt(o.a,3)}}</td>`+
           `<td>${{fmt(o.s,1)}}</td><td>${{o.t||'&mdash;'}}</td><td>${{sym(o)}}</td>`+
           `<td>${{fmt(o.l,4)}}</td></tr>`;
  }}).join('');
}}
function apply(){{
  const s=q.value.trim().toLowerCase();
  let L=s?DATA.filter(o=>o.d.toLowerCase().includes(s)):DATA;
  // sort the whole matching set before trimming to PREVIEW, so "period ascending" means
  // the fastest rotators in the catalog rather than a reshuffle of the same ten rows
  if(sortKey){{
    const k=sortKey;
    L=L.slice().sort((x,y)=>{{
      let a=x[k], b=y[k];
      if(a==null&&b==null) return 0;
      if(a==null) return 1;            // missing values always sort last
      if(b==null) return -1;
      if(typeof a==='string') return sortDir*a.localeCompare(b);
      return sortDir*(a-b);
    }});
  }}
  render(L);
}}
function sortBy(th){{
  const k=th.dataset.k;
  sortDir = (k===sortKey) ? -sortDir : 1;
  sortKey = k;
  document.querySelectorAll('th[data-k]').forEach(h=>h.removeAttribute('aria-sort'));
  th.setAttribute('aria-sort', sortDir>0?'ascending':'descending');
  apply();
}}
document.querySelectorAll('th[data-k]').forEach(th=>{{
  th.addEventListener('click',()=>sortBy(th));
  th.addEventListener('keydown',e=>{{
    if(e.key==='Enter'||e.key===' '){{ e.preventDefault(); sortBy(th); }}
  }});
}});
q.addEventListener('input',apply);
// every object in the index has a page now, so any row is a valid target
document.getElementById('rand').addEventListener('click',()=>{{
  if(!DATA.length) return;
  const o = DATA[Math.floor(Math.random()*DATA.length)];
  window.location.href = `asteroid/${{o.k}}.html`;
}});
fetch(IDX_URL).then(r=>r.json()).then(j=>{{
  DATA=j.sort((a,b)=>(b.g-a.g)||a.d.localeCompare(b.d));   // modelled first
  apply();
}}).catch(()=>{{hits.textContent='could not load the catalog index';}});
</script>
'''
open(f'{ROOT}/index.html', 'w').write(html)
print(f'  index.html {os.path.getsize(f"{ROOT}/index.html")/1024:.0f} KB')
print('  ' + ', '.join(f'{k}={v:,}' if isinstance(v, int) else f'{k}={v:.2f}'
                       for k, v in S.items()))

# ---------------------------------------------------------------- full catalog browser
# Separate page so the landing page stays readable. No row cap here: the whole index is
# already in memory after one fetch, and rendering is windowed so 16,000 rows stay responsive.
search = f'''<!doctype html><meta charset="utf-8">
<title>Browse the catalog — TESSELLATE asteroid rotation rates</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="assets/viewer.css">
<style>
  /* viewer.css pins body to the viewport for the 3D pages; this page must scroll normally,
     and position:sticky needs a real document scroller to anchor to. */
  body{{overflow:visible;height:auto;position:static}}
  main{{max-width:1040px;margin:0 auto;padding:36px 24px 80px}}
  h1{{font-size:24px;margin:0 0 6px}}
  a.home{{display:inline-block;font-size:12px;color:var(--text-dim);text-decoration:none;
    margin-bottom:12px;border-bottom:1px solid transparent}}
  a.home:hover{{color:var(--accent);border-bottom-color:var(--accent)}}
  .toolbar{{position:sticky;top:0;background:var(--bg);padding:12px 0 14px;z-index:5;
    border-bottom:1px solid var(--panel-border);margin-bottom:10px}}
  #q{{width:100%;padding:11px 13px;font-size:14px;border-radius:8px;background:#141a26;
    border:1px solid var(--panel-border);color:var(--text);font-family:inherit}}
  #q:focus{{outline:none;border-color:var(--accent)}}
  .filters{{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 2px;font-size:12px;
    color:var(--text-dim)}}
  .filters label{{display:flex;align-items:center;gap:5px;cursor:pointer}}
  .filters select{{background:#141a26;border:1px solid var(--panel-border);color:var(--text);
    border-radius:6px;padding:3px 6px;font-family:inherit;font-size:12px}}
  .filters select:focus{{outline:none;border-color:var(--accent)}}
  #hits{{font-size:12px;color:var(--text-faint);margin:8px 0 2px}}
  table{{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;
    font-variant-numeric:tabular-nums;margin-top:6px}}
  th{{text-align:right;padding:9px;color:var(--text-faint);font-weight:500;font-size:11.5px;
    letter-spacing:.04em;cursor:pointer;user-select:none;background:var(--bg);
    box-shadow:inset 0 -1px 0 var(--panel-border)}}
  th:hover{{color:var(--accent)}}
  /* arrow space is always reserved, so switching the sorted column does not shift the header */
  th[data-k]{{white-space:nowrap}}
  th[data-k]:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px;
    color:var(--text)}}
  th[data-k]::after{{content:'\\2195';margin-left:5px;opacity:.25;font-size:10px}}
  th[aria-sort="ascending"]::after{{content:'\\2191';opacity:1;color:var(--accent)}}
  th[aria-sort="descending"]::after{{content:'\\2193';opacity:1;color:var(--accent)}}
  th[aria-sort]{{color:var(--text)}}
  td{{text-align:right;padding:6px 9px;border-bottom:1px solid var(--hairline)}}
  /* ---- MOBILE. Same approach as the landing page: the table scrolls in its own container
     and the least essential columns drop, so the page itself never scrolls sideways. */
  @media (max-width:640px){{
    main{{padding:22px 14px 56px}}
    h1{{font-size:21px}}
    .filters{{gap:10px;font-size:11.5px}}
    .tablewrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
    table{{font-size:12px;min-width:460px}}
    th,td{{padding:6px 7px}}
    th:nth-child(4),td:nth-child(4),th:nth-child(7),td:nth-child(7){{display:none}}
    .toolbar{{padding:10px 0 12px}}
  }}
  th:first-child,td:first-child{{text-align:left}}
  tbody tr:hover{{background:#161d2b}}
  td a{{color:var(--accent);text-decoration:none}}
  /* symmetry tags. No yellow; each tint reads on the dark ground and the label carries the
     meaning, so the colour is reinforcement rather than the only encoding. */
  .tag{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;
    letter-spacing:.02em;white-space:nowrap;border:1px solid transparent}}
  .tag.symmetric{{color:#5ec8b4;background:rgba(94,200,180,.11);border-color:rgba(94,200,180,.28)}}
  .tag.asymmetric{{color:#e09a5a;background:rgba(224,154,90,.11);border-color:rgba(224,154,90,.28)}}
  .tag.monomodal{{color:#b294e0;background:rgba(178,148,224,.12);border-color:rgba(178,148,224,.30)}}
  td a:hover{{text-decoration:underline}}
  /* Every object links now. Marking the no-model rows by DIMMING them was a mistake: dim grey
     was previously the not-a-link colour, so a working link looked dead. Keep the link colour
     and carry the distinction in an explicit badge instead. */
  .lcflag{{display:inline-block;margin-left:6px;padding:0 5px;border-radius:4px;font-size:9.5px;
    letter-spacing:.04em;color:var(--text-faint);border:1px solid var(--panel-border);
    vertical-align:1px}}
</style>
<main>
  <a class="home" href="index.html">&larr; Catalog home</a>
  <h1>Browse the catalog</h1>
  <div class="toolbar">
    <input id="q" type="search" placeholder="Search {S['n']:,} objects by name or number&hellip;"
           autocomplete="off">
    <div class="filters">
      <label><input type="checkbox" id="fpage"> Only objects with a shape model</label>
      <label><input type="checkbox" id="fnew"> Only new to the literature</label>
      <label><input type="checkbox" id="flong"> Only longer than 24 hours</label>
      <label>Type
        <select id="ftype">
          <option value="">any</option>
          <option value="*">any classified</option>
          <option value="S">S-complex (stony)</option>
          <option value="C">C-complex (carbonaceous)</option>
          <option value="X">X-complex (metallic / enstatite)</option>
          <option value="D">D / T (primitive, outer belt)</option>
          <option value="V">V (basaltic)</option>
        </select>
      </label>
    </div>
    <div id="hits"></div>
  </div>
  <div class="tablewrap"><table><thead><tr>
    <th data-k="d" tabindex="0" role="button" aria-sort="ascending">Object</th>
    <th data-k="p" tabindex="0" role="button">Period (hours)</th>
    <th data-k="a" tabindex="0" role="button">Amplitude</th>
    <th data-k="s" tabindex="0" role="button">Diameter (km)</th>
    <th data-k="t" tabindex="0" role="button">Type</th>
    <th data-k="m" tabindex="0" role="button" title="+1 symmetric (two identical maxima), 0 equal odd and even power, -1 monomodal (one maximum per rotation)">Symmetry</th>
    <th data-k="l" tabindex="0" role="button">LCDB (hours)</th>
  </tr></thead><tbody id="rows"></tbody></table></div>
</main>
<script>
const rows=document.getElementById('rows'), hits=document.getElementById('hits'),
      q=document.getElementById('q');
let DATA=[], VIEW=[], sortKey='d', sortDir=1, shown=0;
const CHUNK=300;
const fmt=(v,d)=>v==null?'&mdash;':(+v).toFixed(d);
const SYMLBL={{symmetric:'Symmetric',asymmetric:'Asymmetric',monomodal:'Monomodal'}};
function sym(o){{
  if(o.m==null) return '&mdash;';
  const t=` title="${{SYMLBL[o.y]||''}} — signed symmetry ${{o.m}} (odd/even ratio ${{o.o}})"`;
  return `<span class="tag ${{o.y}}"${{t}}>${{o.m>0?'+':''}}${{o.m.toFixed(2)}}</span>`;
}}
function row(o){{
  const name=`<a href="asteroid/${{o.k}}.html">${{o.d}}</a>`+(o.g?'':'<span class="lcflag" title="Lightcurve only &mdash; no shape model for this object">lc</span>');
  return `<tr><td>${{name}}</td><td>${{fmt(o.p,4)}}</td><td>${{fmt(o.a,3)}}</td>`+
         `<td>${{fmt(o.s,1)}}</td><td>${{o.t||'&mdash;'}}</td><td>${{sym(o)}}</td>`+
         `<td>${{fmt(o.l,4)}}</td></tr>`;
}}
// windowed rendering: the full list stays in memory, the DOM only grows as you scroll,
// so 16,000 rows do not have to be laid out at once
function draw(reset){{
  if(reset){{ rows.innerHTML=''; shown=0; }}
  const next=VIEW.slice(shown, shown+CHUNK);
  rows.insertAdjacentHTML('beforeend', next.map(row).join(''));
  shown+=next.length;
  hits.textContent=`${{VIEW.length.toLocaleString()}} of ${{DATA.length.toLocaleString()}} objects`
    + (shown<VIEW.length?` — ${{shown.toLocaleString()}} loaded, scroll for more`:'');
}}
function apply(){{
  const s=q.value.trim().toLowerCase();
  let L=DATA;
  if(s) L=L.filter(o=>o.d.toLowerCase().includes(s));
  if(document.getElementById('fpage').checked) L=L.filter(o=>o.g);
  if(document.getElementById('fnew').checked)  L=L.filter(o=>o.l==null);
  if(document.getElementById('flong').checked) L=L.filter(o=>o.p>24);
  const ty=document.getElementById('ftype').value;
  if(ty==='*') L=L.filter(o=>o.c);            // any classified object
  else if(ty) L=L.filter(o=>o.c===ty);
  const k=sortKey;
  L=L.slice().sort((x,y)=>{{
    let a=x[k], b=y[k];
    if(a==null&&b==null) return 0;
    if(a==null) return 1;            // missing values always sort last
    if(b==null) return -1;
    if(typeof a==='string') return sortDir*a.localeCompare(b);
    return sortDir*(a-b);
  }});
  VIEW=L; draw(true);
}}
q.addEventListener('input',apply);
for(const id of ['fpage','fnew','flong','ftype'])
  document.getElementById(id).addEventListener('change',apply);
function sortBy(th){{
  const k=th.dataset.k;
  sortDir = (k===sortKey) ? -sortDir : 1;
  sortKey = k;
  document.querySelectorAll('th[data-k]').forEach(h=>h.removeAttribute('aria-sort'));
  th.setAttribute('aria-sort', sortDir>0?'ascending':'descending');
  apply();
  window.scrollTo({{top:0}});      // a re-sort with the list scrolled down is disorienting
}}
document.querySelectorAll('th[data-k]').forEach(th=>{{
  th.addEventListener('click',()=>sortBy(th));
  th.addEventListener('keydown',e=>{{
    if(e.key==='Enter'||e.key===' '){{ e.preventDefault(); sortBy(th); }}
  }});
}});
window.addEventListener('scroll',()=>{{
  if(shown<VIEW.length && window.innerHeight+window.scrollY > document.body.offsetHeight-600)
    draw(false);
}});
// Remember the browse state, so coming back from an asteroid page returns you to the list you
// had, not a blank one. sessionStorage: per-tab, cleared when the tab closes, never sent anywhere.
const SKEY='tessellate.browse';
function saveState(){{
  try{{ sessionStorage.setItem(SKEY, JSON.stringify({{
    q:q.value, sortKey, sortDir, y:window.scrollY,
    fpage:document.getElementById('fpage').checked,
    fnew:document.getElementById('fnew').checked,
    flong:document.getElementById('flong').checked,
    ftype:document.getElementById('ftype').value }})); }}catch(e){{}}
}}
function restoreState(){{
  try{{
    const v=JSON.parse(sessionStorage.getItem(SKEY)||'null'); if(!v) return null;
    q.value=v.q||'';
    document.getElementById('fpage').checked=!!v.fpage;
    document.getElementById('fnew').checked=!!v.fnew;
    document.getElementById('flong').checked=!!v.flong;
    document.getElementById('ftype').value=v.ftype||'';
    if(v.sortKey){{
      sortKey=v.sortKey; sortDir=v.sortDir||1;
      document.querySelectorAll('th[data-k]').forEach(h=>h.removeAttribute('aria-sort'));
      const th=document.querySelector(`th[data-k="${{sortKey}}"]`);
      if(th) th.setAttribute('aria-sort', sortDir>0?'ascending':'descending');
    }}
    return v;
  }}catch(e){{ return null; }}
}}
window.addEventListener('pagehide', saveState);
document.addEventListener('click', e=>{{ if(e.target.closest('#rows a')) saveState(); }});

fetch('data/index.json?v={IDXV}').then(r=>r.json()).then(j=>{{
  DATA=j;
  const v=restoreState();
  apply();
  if(v && v.y){{
    // the list renders in chunks, so keep loading until the saved offset is reachable
    let guard=0;
    (function seek(){{
      if(window.scrollY<v.y && shown<VIEW.length && guard++<200){{ draw(false); }}
      if(document.body.offsetHeight>=v.y+window.innerHeight || shown>=VIEW.length){{
        window.scrollTo(0,v.y); return;
      }}
      if(guard<200) requestAnimationFrame(seek); else window.scrollTo(0,v.y);
    }})();
  }}
}}).catch(()=>{{hits.textContent='Could not load the catalog index';}});
</script>
'''
open(f'{ROOT}/search.html', 'w').write(search)
print(f'  search.html {os.path.getsize(f"{ROOT}/search.html")/1024:.0f} KB')
