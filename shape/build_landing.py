"""Build the public landing page: headline figures, live search, and the catalog index.

SEARCH DESIGN. 16,427 objects is too many to ship as one blob and too few to justify a backend.
The whole index is ~600 KB of JSON (designation, period, amplitude, size, flags) which is a
single cached fetch, and filtering happens client-side. That keeps the site static -- no server,
nothing to break -- while still giving instant search over the full catalog.

Only objects that actually have a page are linked. The rest of the catalog is searchable and
its numbers are shown, but a link is only rendered where a shape model exists, so nobody follows
a dead URL.
"""
import json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')
CAT = f'{Y34}/population_figs/all_sector_report_v5_doubled.csv'
FIGS = f'{Y34}/population_figs/site'


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

have_page = {f[:-5] for f in os.listdir(f'{ROOT}/asteroid')} if os.path.isdir(f'{ROOT}/asteroid') else set()
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
            g=bool(row.has_page))
       for row in r.itertuples()]
json.dump(idx, open(f'{ROOT}/data/index.json', 'w'), separators=(',', ':'))
print(f'  index.json {os.path.getsize(f"{ROOT}/data/index.json")/1024:.0f} KB')

S = dict(n=len(r), new=int(r.published_rot_per_hr.isna().sum()),
         lcdb=int(r.published_rot_per_hr.notna().sum()),
         med=r.period_hr.median(), long=int((r.period_hr > 24).sum()),
         base=float((r.baseline_hr / 24).median()),
         rots=int((r.baseline_hr / r.period_hr).median()),
         pts=int(r.n_points.median()),
         pages=int(r.has_page.sum()))

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
  .figs figure{{margin:0;background:var(--panel);border:1px solid var(--panel-border);
    border-radius:9px;padding:10px}}
  .figs img{{width:100%;height:auto;border-radius:5px;display:block;background:#fff}}
  .figs figcaption{{font-size:11.5px;color:var(--text-dim);margin-top:7px;line-height:1.45}}
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
  td a:hover{{text-decoration:underline}}
  .nopage{{color:var(--text-dim)}}
  a.more{{color:var(--accent);text-decoration:none;font-weight:600}}
  a.more:hover{{text-decoration:underline}}
  .caveat{{margin-top:30px;padding:13px 15px;border-radius:8px;background:#e0a4581a;
    border:1px solid #e0a45840;font-size:12.5px;line-height:1.6;color:#e8c99a;max-width:72ch}}
</style>
<main>
  <p class="eyebrow">TESS sectors 27&ndash;55 &middot; TESSELLATE</p>
  <h1>Asteroid rotation catalog</h1>
  <p class="lede">Rotation periods, phase-folded lightcurves and convex shape models for
     asteroids observed by TESS. Because TESS watches the same field continuously for about a
     month, these lightcurves run unbroken for weeks &mdash; reaching slow rotators that a single
     night cannot close, and pinning periods across hundreds of consecutive rotations.</p>

  <div class="kpi">
    <div><div class="v">{S['n']:,}</div><div class="l">rotation periods</div></div>
    <div><div class="v">{S['new']:,}</div><div class="l">with no previously published period</div></div>
    <div><div class="v">{S['long']:,}</div><div class="l">longer than 24 hours</div></div>
    <div><div class="v">{S['base']:.0f} days</div><div class="l">median observing baseline</div></div>
    <div><div class="v">{S['pts']:,}</div><div class="l">median measurements each</div></div>
  </div>

  <h2>Search the catalog</h2>
  <p class="note">Type a name or number for a quick look. Shape-model pages exist for
     {S['pages']:,} objects so far and are linked; the rest are listed but not yet published.</p>
  <input id="q" type="search" placeholder="e.g. Eurydike, 3550, Link&hellip;" autocomplete="off">
  <div id="hits"></div>
  <table><thead><tr>
    <th data-k="d" tabindex="0" role="button">Object</th>
    <th data-k="p" tabindex="0" role="button">Period (hours)</th>
    <th data-k="a" tabindex="0" role="button">Amplitude</th>
    <th data-k="s" tabindex="0" role="button">Diameter (km)</th>
    <th data-k="t" tabindex="0" role="button">Type</th>
    <th data-k="l" tabindex="0" role="button">LCDB (hours)</th>
  </tr></thead><tbody id="rows"></tbody></table>
  <p class="note" style="margin-top:12px"><a class="more" href="search.html">Browse the full
     catalog &rarr;</a> &mdash; all {S['n']:,} objects, with filters and sorting.</p>

  <h2>The population</h2>
  <div class="figs">
    <figure><img src="assets/period_hist.png" alt="Distribution of rotation periods">
      <figcaption>Rotation periods. {S['long']:,} objects turn more slowly than once a day,
      the regime ground-based photometry struggles to close.</figcaption></figure>
    <figure><img src="assets/spin_size.png" alt="Spin rate against diameter">
      <figcaption>Spin rate against size. The 2.2 hour barrier is where a loosely bound rubble pile
      would fly apart; almost nothing sits above it.</figcaption></figure>
    <figure><img src="assets/amplitude_period.png" alt="Lightcurve amplitude against period">
      <figcaption>Amplitude against period. Larger amplitude means a more elongated body, since
      a rounder one varies less as it turns.</figcaption></figure>
    <figure style="grid-column:1/-1"><img src="assets/orbital_elements.png"
        alt="Semi-major axis against inclination and eccentricity">
      <figcaption>Where these asteroids orbit. The dashed lines are Jupiter mean-motion
      resonances, which clear the Kirkwood gaps; clumps are collisional families. The catalog
      samples the inner, middle and outer belt about equally.</figcaption></figure>
  </div>

  <h2>What you can download</h2>
  <p class="note">Each object page carries its phase-folded lightcurve with per-bin
     uncertainties, the full stacked photometry behind it, and a rotatable convex shape model
     &mdash; all downloadable, and the shape model is synchronised to the lightcurve so you can
     watch which face produces which feature.</p>

  <div class="caveat">
    <strong>Reading a shape model.</strong> Most of these asteroids were seen during a single
    apparition, from one direction. That constrains the shape but not its orientation in space,
    so the spin axis is assumed rather than measured and every page says so. Convex inversion
    also cannot represent concavities, so models systematically under-reach the deepest minima.
    The rotation periods, by contrast, are well determined.
  </div>
</main>
<script>
const IDX_URL='data/index.json';
const rows=document.getElementById('rows'), hits=document.getElementById('hits'),
      q=document.getElementById('q');
let DATA=[], sortKey=null, sortDir=1;
const fmt=(v,d)=>v==null?'&mdash;':(+v).toFixed(d);
const PREVIEW=10;
const LABEL={{d:'name',p:'period',a:'amplitude',s:'diameter',t:'type',l:'LCDB period'}};
function render(list){{
  let note='';
  if(list.length>PREVIEW)
    note = sortKey ? ` — showing the ${{PREVIEW}} ${{sortDir>0?'lowest':'highest'}} by `
                     + LABEL[sortKey]
                   : ` — showing ${{PREVIEW}}`;
  hits.textContent=`${{list.length.toLocaleString()}} of ${{DATA.length.toLocaleString()}} objects`+note;
  rows.innerHTML=list.slice(0,PREVIEW).map(o=>{{
    const name=o.g?`<a href="asteroid/${{o.k}}.html">${{o.d}}</a>`
                  :`<span class="nopage">${{o.d}}</span>`;
    return `<tr><td>${{name}}</td><td>${{fmt(o.p,4)}}</td><td>${{fmt(o.a,3)}}</td>`+
           `<td>${{fmt(o.s,1)}}</td><td>${{o.t||'&mdash;'}}</td><td>${{fmt(o.l,4)}}</td></tr>`;
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
fetch(IDX_URL).then(r=>r.json()).then(j=>{{
  DATA=j.sort((a,b)=>(b.g-a.g)||a.d.localeCompare(b.d));
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
  th:first-child,td:first-child{{text-align:left}}
  tbody tr:hover{{background:#161d2b}}
  td a{{color:var(--accent);text-decoration:none}}
  td a:hover{{text-decoration:underline}}
  .nopage{{color:var(--text-dim)}}
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
  <table><thead><tr>
    <th data-k="d" tabindex="0" role="button" aria-sort="ascending">Object</th>
    <th data-k="p" tabindex="0" role="button">Period (hours)</th>
    <th data-k="a" tabindex="0" role="button">Amplitude</th>
    <th data-k="s" tabindex="0" role="button">Diameter (km)</th>
    <th data-k="t" tabindex="0" role="button">Type</th>
    <th data-k="l" tabindex="0" role="button">LCDB (hours)</th>
  </tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
const rows=document.getElementById('rows'), hits=document.getElementById('hits'),
      q=document.getElementById('q');
let DATA=[], VIEW=[], sortKey='d', sortDir=1, shown=0;
const CHUNK=300;
const fmt=(v,d)=>v==null?'&mdash;':(+v).toFixed(d);
function row(o){{
  const name=o.g?`<a href="asteroid/${{o.k}}.html">${{o.d}}</a>`
                :`<span class="nopage">${{o.d}}</span>`;
  return `<tr><td>${{name}}</td><td>${{fmt(o.p,4)}}</td><td>${{fmt(o.a,3)}}</td>`+
         `<td>${{fmt(o.s,1)}}</td><td>${{o.t||'&mdash;'}}</td><td>${{fmt(o.l,4)}}</td></tr>`;
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
fetch('data/index.json').then(r=>r.json()).then(j=>{{
  DATA=j; apply();
}}).catch(()=>{{hits.textContent='Could not load the catalog index';}});
</script>
'''
open(f'{ROOT}/search.html', 'w').write(search)
print(f'  search.html {os.path.getsize(f"{ROOT}/search.html")/1024:.0f} KB')
