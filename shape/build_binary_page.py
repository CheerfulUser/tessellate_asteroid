"""Binary / multiple asteroid page for the site. Takes the site's normal page and data for one object
and the TESS fits from periods/known_binary_stack.py:
  orbits/<key>.json         fitorb: the orbital period fitted to TESS, two uniform discs (primary radius 1, secondary
                            q) on a circular orbit of radius a, impact parameter b, the second event scaled by k2
                            (shadows and unequal albedo, not modelled, make the depths differ); its stack at the fit
  orbits/<key>_check3.json  check3: a further periodic signal after the primary and the events are removed (optional)
and writes data/binaries/<key>.json (a list of companions and a list of further signals, so any number of each) plus
asteroid/<key>.html using assets/viewer_binary.js.

Density per companion, from Kepler's third law with both bodies of one bulk density:
  rho = 24 pi (a/Dp)^3 / (G P^2 (1 + q^3))
(independent of the absolute size). Uncertainty propagated from the fitted a, q and P. The uniform-disc model ignores
shadows, which lengthen events at non-zero phase angle; longer events need a tighter orbit, so the density is then
UNDER-estimated. It is shown with that caveat.

    python3 shape/build_binary_page.py 1830_Pogson "(1830) Pogson" 24.24 3.2627
"""
import json
import os
import re
import sys

import numpy as np

SITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the site repository root
HERE = SITE
KB = "/Users/rridden/Documents/work/code/tess/asteroid/y3_4/label_batches/known_binary_stacks"
G = 6.674e-11

key, des, p_orb_lcdb, p_add_lcdb = sys.argv[1], sys.argv[2], float(sys.argv[3]), (float(sys.argv[4]) if len(sys.argv) > 4 else None)
# start from the NORMAL page as committed (the file on disk may already be the binary version)
import subprocess
html = subprocess.run(["git", "-C", SITE, "show", f"HEAD:asteroid/{key}.html"], capture_output=True, text=True, check=True).stdout
if "viewer_binary.js" in html:
    raise SystemExit(f"HEAD:asteroid/{key}.html is already a binary page; build from the commit before it")
P1 = float(re.search(r"Rotation period</span><span class=\"stat-value\">([0-9.]+) hours", html).group(1))


def overlap(d, r1, r2):
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    full = d <= abs(r1 - r2)
    out[full] = np.pi * min(r1, r2) ** 2
    part = (d < r1 + r2) & ~full
    dd = d[part]
    a1 = r1 ** 2 * np.arccos(np.clip((dd ** 2 + r1 ** 2 - r2 ** 2) / (2 * dd * r1), -1, 1))
    a2 = r2 ** 2 * np.arccos(np.clip((dd ** 2 + r2 ** 2 - r1 ** 2) / (2 * dd * r2), -1, 1))
    a3 = 0.5 * np.sqrt(np.clip((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2), 0, None))
    out[part] = a1 + a2 - a3
    return out


def events(ph, off, q, a, b, k2):
    """The two-disc event model against orbital phase (secondary in front at phase 0)."""
    th = 2 * np.pi * ph
    d = np.sqrt((a * np.sin(th)) ** 2 + (b * np.cos(th)) ** 2)
    loss = overlap(d, 1.0, q) / (np.pi * (1 + q ** 2))
    return off - np.where(np.cos(th) > 0, loss, k2 * loss)


def density(a_dp, q, P_hr, a_dp_err, q_err, P_err):
    P = P_hr * 3600
    rho = 24 * np.pi * a_dp ** 3 / (G * P ** 2 * (1 + q ** 3)) / 1000          # g/cm^3
    rel = np.sqrt((3 * a_dp_err / a_dp) ** 2 + (2 * P_err / P_hr) ** 2 + (3 * q ** 2 * q_err / (1 + q ** 3)) ** 2)
    return rho, rho * rel


o = json.load(open(f"{KB}/orbits/{key}.json"))
c, m, e = np.array(o["stack_phase"]), np.array(o["stack_mean"]), np.array(o["stack_err"])
srt = np.argsort(c)
c, m, e = c[srt], m[srt], e[srt]
grid = np.linspace(0, 1, 401)
a_dp, a_dp_err = o["a_Rp"] / 2, o["a_err"] / 2
rho, rho_err = density(a_dp, o["q"], o["P_orb_hr"], a_dp_err, o["q_err"], o["P_orb_err_hr"])
comp = dict(label="Secondary events", P_orb_hr=o["P_orb_hr"], P_orb_err_hr=o["P_orb_err_hr"], q=o["q"], a_Dp=a_dp,
            b_Rp=o["b_Rp"], k2=o["k2"], dip_phase_front=0.0, t0_hr=o["t0_hr"], sec_axes=[1, 1, 1],
            model_phase=grid.tolist(), model=events(grid, o["offset"], o["q"], o["a_Rp"], o["b_Rp"], o["k2"]).tolist(),
            ev_phase=c.tolist(), ev_mean=np.where(e > 0, m, 0).tolist(), ev_err=e.tolist(), chi2r=o["chi2r"],
            density=rho, density_err=rho_err)
signals = []
f3 = f"{KB}/orbits/{key}_check3.json"
if os.path.exists(f3):
    s3 = json.load(open(f3))
    # the further signal is the rotation of the next body (Tertiary when there is one companion)
    signals.append(dict(label="Tertiary rotation", period_hr=s3["P3_hr"], period_err_hr=s3["P3_err_hr"],
                        phase=s3["fold_phase"], mean=s3["fold_mean"], err=s3["fold_err"], model_phase=s3["model_phase"],
                        model=s3["model"], amp=s3["P3_amp"]))
# a body known only from a further signal (its rotation): no orbit, no events, so it is drawn schematically. Its size
# has a floor: the signal's amplitude A_obs (relative to the whole system) = f * A_int, f its share of the light,
# A_int its own peak-to-peak amplitude; with A_int <= A_INT_MAX, f >= A_obs / A_INT_MAX, and with
# f = s^2 / (1 + q^2 + s^2) (s = D3/Dp, same albedo) this gives s_min
A_INT_MAX = 1.0
unconstrained = []
for s_ in signals:
    fmin = s_["amp"] / A_INT_MAX
    s_min = float(np.sqrt(fmin * (1 + o["q"] ** 2) / (1 - fmin)))
    unconstrained.append(dict(label=f"Body spinning in {s_['period_hr']:.4f} h", P_spin_hr=s_["period_hr"], size_ratio_min=s_min,
                              note="orbit, distance and size not measured; drawn at its minimum size at a placeholder position"))
b = dict(P1_hr=P1, companions=[comp], signals=signals, unconstrained=unconstrained,
         notes="mock-up: companions from the two-disc fit to TESS; further signals after the primary and the events are "
               "removed; secondary shapes, shadows and the rotation-orbit phase are not modelled")
os.makedirs(f"{HERE}/data/binaries", exist_ok=True)
json.dump(b, open(f"{HERE}/data/binaries/{key}.json", "w"))

# Info panel: the normal page layout, with the primary's physical properties carrying the system's bulk density, then
# one section per further body. Diameters: the catalogue diameter is the whole system's effective diameter,
# D_eff^2 = Dp^2 + Ds^2 (+ D3^2), so the primary's own is D_eff / sqrt(1 + q^2); a further body's minimum size is not
# included (it is only a lower bound). The scale bars use the primary's diameter.
def row(k, v):
    return f'<div class="stat-row"><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>'
D_eff = float(re.search(r'"diam_km":([0-9.]+)', html).group(1))
Dp = D_eff / np.sqrt(1 + o["q"] ** 2)
Ds = o["q"] * Dp
html = re.sub(r'(<span class="stat-label">Diameter</span><span class="stat-value">)[^<]*(</span>)',
              lambda mm: mm.group(1) + f"{Dp:.1f} km" + mm.group(2), html, count=1)
# bulk density in the TESSELLATE solution, right after the rotation period (it comes from the TESS fit)
html = re.sub(r'(<span class="stat-label">Rotation period</span><span class="stat-value">[^<]*</span></div>)',
              lambda mm: mm.group(1) + row("Bulk density", f"{rho:.2f} &plusmn; {rho_err:.2f} g/cm<sup>3</sup>"), html, count=1)
# a tag under the name: how many bodies the system has (primary + companions + bodies known only from a signal)
n_bodies = 1 + 1 + len(unconstrained)
tag = ('<p style="margin:2px 0 10px"><span style="display:inline-block;font-family:\'IBM Plex Mono\',ui-monospace,monospace;'
       'font-size:11px;letter-spacing:.06em;padding:2px 8px;border-radius:10px;border:1px solid var(--accent);'
       f'color:var(--accent);background:var(--accent-bg)">{n_bodies} bodies</span></p>')
html = re.sub(r'(<h1>[^<]*</h1>)', lambda mm: mm.group(1) + tag, html, count=1)
html = html.replace('"diam_km":' + re.search(r'"diam_km":([0-9.]+)', html).group(1), f'"diam_km":{Dp:.3f}', 1)
names = ["Secondary", "Tertiary", "Quaternary"]
sec = (f'<p class="sec">{names[0]}</p>'
       + row("Diameter", f"{Ds:.1f} km")
       + row("Orbit period", f"{o['P_orb_hr']:.3f} &plusmn; {o['P_orb_err_hr']:.3f} h")
       + row("Separation", f"{a_dp * Dp:.1f} km")
       + row("Inclination", f"{np.degrees(np.arccos(min(o['b_Rp'] / o['a_Rp'], 1))):.1f}&deg;"))
for k_, u_ in enumerate(unconstrained):
    sec += (f'<p class="sec">{names[k_ + 1]}</p>'
            + row("Diameter", f"&ge; {u_['size_ratio_min'] * Dp:.1f} km")
            + row("Rotation period", f"{u_['P_spin_hr']:.4f} h")
            + row("Orbit", "not measured"))
html = html.replace('<p class="sec">Downloads</p>', sec + '<p class="sec">Downloads</p>', 1)
html = re.sub(r'<script src="\.\./assets/viewer\.js[^"]*"></script>',
              f'<script src="../assets/viewer_binary.js?v={int(os.path.getmtime(os.path.join(HERE, "assets", "viewer_binary.js")))}"></script>', html)
html = html.replace('"diam_km":', f'"binary":"../data/binaries/{key}.json","diam_km":', 1)
html = html.replace('href="../assets/viewer.css?v=', 'href="../assets/viewer.css?x=')
open(f"{HERE}/asteroid/{key}.html", "w").write(html)
print(f"{key}: P1 {P1} h; companion 1: orbit {o['P_orb_hr']:.4f} h, q {o['q']:.3f}, a/Dp {a_dp:.2f}, b {o['b_Rp']:.2f}, "
      f"density {rho:.2f} +- {rho_err:.2f} g/cm3; further signals {[round(s_['period_hr'], 4) for s_ in signals]}")
