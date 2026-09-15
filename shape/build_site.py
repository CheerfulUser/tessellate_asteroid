"""Generate the public site by SPLITTING the known-good rendered page, not by re-authoring it.

WHY THIS SHAPE. Embedding the mesh in every page gives 73 KB per object; 16,000 of those is
1.19 GB and GitHub Pages enforces a hard 1 GB limit. So the page has to be split from its data.

But the first attempt at that split re-authored the CSS and markup from build_shape_viewer.py's
f-string SOURCE, which still contained `{{`/`}}` escapes -- the whole stylesheet came out invalid
and the page rendered as unstyled wreckage. The lesson: split the RENDERED output, where the
escapes are already resolved and the result is a page that has been checked by eye.

So this runs the existing generator unchanged, then slices its output:

    <style> ... </style>    -> web/assets/viewer.css   identical for every object, written once
    <script> ... </script>  -> web/assets/viewer.js    likewise, with the embedded `const D = {...}`
                                                       swapped for a fetch of the external JSON
    everything else         -> web/asteroid/KEY.html   the per-object shell, ~4 KB

CSS and JS are byte-identical across objects -- only the data differs -- so the appearance is
exactly what build_shape_viewer.py produced. The shell keeps the real markup rather than
rebuilding it in JS, which is what lost the physical-properties rows last time.

Usage:  python shape/build_site.py eurydike deflotte
"""
import gzip, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get('SHAPE_SRC',
                     '/Users/rridden/Documents/work/code/tess/tessellate_test/asteroid_dev/shape_modeling')
WEB, DATA = ROOT, f'{ROOT}/data'
COORD_DP = 4
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def render_page(tag):
    """Run the existing generator and return the page it produces."""
    out = f'{SRC}/{tag}_shape_viewer.html'
    subprocess.run([sys.executable, f'{SRC}/build_shape_viewer.py', tag],
                   cwd=SRC, check=True, capture_output=True)
    return open(out).read()


def split(html):
    css = html[html.index('<style>') + 7:html.index('</style>')]
    js = html[html.index('<script>') + 8:html.rindex('</script>')]
    body = (html[:html.index('<style>')].replace('<title>', '<!--title-->', 1)
            + html[html.index('</style>') + 8:html.index('<script>')])
    return css.strip(), js, body


def js_to_shared(js):
    """Replace the embedded data blob with a fetch, and defer startup until it resolves.
    Everything else in the script is untouched."""
    m = re.search(r'^const D = \{.*?\};\s*$', js, re.M | re.S)
    if not m:
        raise SystemExit('could not find the embedded `const D = {...};` to replace')
    head = ('// Shared by every per-asteroid page. The mesh and folded lightcurve are FETCHED\n'
            '// rather than embedded: 73 KB per page x 16,000 objects exceeds the 1 GB GitHub\n'
            '// Pages limit. Everything below the fetch is unchanged from the single-page viewer.\n'
            'let D;\n'
            'async function boot(){\n'
            '  const A = window.AST;\n'
            '  const [sh, lc] = await Promise.all([\n'
            '    fetch(A.shape).then(r=>r.json()),\n'
            '    A.has_lc ? fetch(A.lc).then(r=>r.json()).catch(()=>null) : Promise.resolve(null)\n'
            '  ]);\n'
            '  const verts=[]; for(let i=0;i<sh.v.length;i+=3) verts.push([sh.v[i],sh.v[i+1],sh.v[i+2]]);\n'
            '  // faces are polygons of varying size, so walk the per-face counts rather than step by 3\n'
            '  const facets=[]; let _o=0;\n'
            '  for(const k of sh.fn){ facets.push(sh.f.slice(_o,_o+k)); _o+=k; }\n'
            '  D = Object.assign({}, A.D, {verts, facets,\n'
            '        lcP: lc? lc.phase:[], lcF: lc? lc.flux:[],\n'
            '        lcE: lc? (lc.err||null):null, lcM: lc? (lc.model_flux||null):null});\n'
            '  run();\n'
            '}\n'
            'function run(){\n')
    return head + js[m.end():] + '\n}\nboot();\n'


def compact_mesh(mesh):
    """Flat arrays at 4 dp: 2.4x smaller than nested lists at full float repr, and far finer
    than a single-apparition convex hull is determined to.

    Faces are NOT triangles. A convex hull built from 289 face normals has 289 POLYGONAL faces
    -- for Eurydike they run 4 to 9 vertices each. An earlier version flattened the index list
    and the reader regrouped it in threes, turning 289 correct polygons into 574 triangles
    stitched across unrelated faces. So the per-face vertex COUNTS travel with the indices."""
    v, f = mesh['recovered']['verts'], mesh['recovered']['facets']
    return {'v': [round(float(c), COORD_DP) for p in v for c in p],
            'f': [int(i) for t in f for i in t],
            'fn': [len(t) for t in f]}


def build(tag, assets_written=[False]):
    html = render_page(tag)
    css, js, body = split(html)
    res = json.load(open(f'{SRC}/{tag}_results.json'))
    key = sn(res['target'])
    for d in ('assets',):
        os.makedirs(f'{WEB}/{d}', exist_ok=True)
    os.makedirs(f'{WEB}/asteroid', exist_ok=True)
    os.makedirs(f'{DATA}/shapes', exist_ok=True)
    os.makedirs(f'{DATA}/lightcurves', exist_ok=True)

    if not assets_written[0]:
        extra = ('\n.dl{display:flex;flex-direction:column;gap:4px;margin:2px 0 4px}\n'
                 '.dl a{font-size:11.5px;color:var(--accent);text-decoration:none;'
                 'border-bottom:1px solid transparent}\n'
                 '.dl a:hover{border-bottom-color:var(--accent)}\n'
                 '.dl a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}\n')
        open(f'{WEB}/assets/viewer.css', 'w').write(css + extra)
        open(f'{WEB}/assets/viewer.js', 'w').write(js_to_shared(js))
        assets_written[0] = True

    # the display config that lived inside the embedded blob, minus the bulky arrays
    dm = re.search(r'^const D = (\{.*?\});\s*$', js, re.M | re.S).group(1)
    d0 = json.loads(dm)
    keep = {k: d0[k] for k in ('alb', 'weak', 'lo', 'hi', 'lut', 'camQ', 'sunCam', 'aspect',
                               'geoAlbedo', 'tint', 'spec') if k in d0}

    json.dump(compact_mesh(json.load(open(f'{SRC}/{tag}_mesh_data.json'))),
              open(f'{DATA}/shapes/{key}.json', 'w'), separators=(',', ':'))
    lcj = dict(phase=d0.get('lcP', []), flux=d0.get('lcF', []),
               err=d0.get('lcE'), model_flux=d0.get('lcM'))
    json.dump(lcj, open(f'{DATA}/lightcurves/{key}.json', 'w'), separators=(',', ':'))

    # full-resolution stacked photometry, gzipped: the page plots the binned fold, but anyone
    # refitting needs the points. ~4x smaller than the raw CSV and the browser decompresses it.
    full = f'{Y34}/lcdb_lcs/stacked/{key}_lc.csv'
    full_gz = f'{DATA}/lightcurves/{key}_stacked.csv.gz'
    has_full = os.path.exists(full)
    if has_full:
        with open(full, 'rb') as fi, gzip.open(full_gz, 'wb', compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)

    dl = ['<p class="sec">Downloads</p><div class="dl">',
          f'<a href="../data/shapes/{key}.json" download>shape model (JSON mesh)</a>',
          f'<a href="../data/lightcurves/{key}.json" download>folded lightcurve (JSON)</a>']
    if has_full:
        dl.append(f'<a href="../data/lightcurves/{key}_stacked.csv.gz" download>'
                  f'full stacked lightcurve (CSV.gz, {os.path.getsize(full_gz)//1024} KB)</a>')
    dl.append('</div>')
    body = body.replace('<div class="caveat">', ''.join(dl) + '<div class="caveat">', 1)

    meta = {'key': key, 'has_lc': bool(lcj['phase']),
            'shape': f'../data/shapes/{key}.json',
            'lc': f'../data/lightcurves/{key}.json', 'D': keep}
    shell = (f'<!doctype html><meta charset="utf-8">\n'
             f'<title>{res["target"]} — TESSELLATE asteroid catalogue</title>\n'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
             f'<link rel="stylesheet" href="../assets/viewer.css">\n'
             + body.replace('<!--title-->', '<!-- ').replace('</title>', ' -->', 1)
             + f'\n<script>window.AST={json.dumps(meta, separators=(",", ":"))};</script>\n'
               f'<script src="../assets/viewer.js"></script>\n')
    open(f'{WEB}/asteroid/{key}.html', 'w').write(shell)
    return dict(key=key, shell=len(shell),
                mesh=os.path.getsize(f'{DATA}/shapes/{key}.json'),
                lc=os.path.getsize(f'{DATA}/lightcurves/{key}.json'))


if __name__ == '__main__':
    rows = [build(t) for t in sys.argv[1:]]
    for r in rows:
        print(f'  {r["key"]:22s} shell {r["shell"]/1024:5.1f} KB  mesh {r["mesh"]/1024:5.1f} KB  '
              f'lc {r["lc"]/1024:4.1f} KB')
    if rows:
        a = sum(os.path.getsize(f'{WEB}/assets/{n}') for n in ('viewer.css', 'viewer.js'))
        per = sum(r['shell'] + r['mesh'] + r['lc'] for r in rows) / len(rows)
        print(f'\nshared assets {a/1024:.1f} KB (cached once)')
        print(f'mean {per/1024:.1f} KB per asteroid -> {per*16000/1e6:.0f} MB at 16,000 '
              f'(embedded would be {73*1024*16000/1e6:.0f} MB; limit 1000 MB)')
