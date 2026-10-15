"""Find the source of the 1.5e-3 difference between our bright.c reimplementation and convexinv's
own TESS model (iter_test, Loreley pole 1), using the exact lcs values convexinv read."""
import sys
import numpy as np
sys.path.insert(0, "/fred/oz335/rridden/asteroids/combined_tess_atlas/code")
import combined_tess_atlas as ct

W = "/fred/oz335/rridden/asteroids/combined_tess_atlas/out/iter_test/work_165_Loreley"
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tok = open(f"{W}/it{k}_p.txt").read().split()
ctl = open(f"{W}/it{k}_c.txt").read().split("\n")
lam, bet = float(ctl[0].split()[0]), float(ctl[1].split()[0])
rows, sizes = [], []
with open(f"{W}/lcs_tess.txt") as f:
    for _ in range(int(f.readline())):
        n = int(f.readline().split()[0]); sizes.append(n)
        rows += [list(map(float, f.readline().split())) for _ in range(n)]
rows = np.array(rows)
jd, sv, ev = rows[:, 0], rows[:, 2:5], rows[:, 5:8]
theirs = np.loadtxt(f"{W}/it{k}_f.txt")
edges = np.cumsum([0] + sizes)

def dev(m):
    return max(np.max(np.abs(m[a:b] / m[a:b].mean() - theirs[a:b] / theirs[a:b].mean())) for a, b in zip(edges[:-1], edges[1:]))

per, jd0, phi0 = float(tok[2]), float(tok[3]), np.radians(float(tok[4]))
a_, d_, k_ = map(float, tok[5:8])
eo, es = ct.body_frame(ev, lam, bet), ct.body_frame(sv, lam, bet)
alpha = np.arccos(np.clip(np.sum(eo * es, 1), -1, 1))
ph = 1 + a_ * np.exp(-alpha / d_) + k_ * alpha
t = open(f"{W}/it{k}_s.txt").read().split(); n = int(t[0])
v = np.array(t[1:1 + 4 * n], float).reshape(n, 4)
for name, per_use in (("par period", per),):
    phi = phi0 + 2 * np.pi * (jd - jd0) / (per_use / 24)
    A, N = v[:-1, 0], v[:-1, 1:]
    base = ct.brightness(A, N, eo, es, phi)
    print("baseline", dev(base * ph))
    print("no phase factor", dev(base))
    print("with dark facet", dev(ct.brightness(v[:, 0], v[:, 1:], eo, es, phi) * ph))
    print("rotation sign flipped", dev(ct.brightness(A, N, eo, es, -phi) * ph))
    print("lambda rounded as par prints", dev(ct.brightness(A, N, ct.body_frame(ev, float(tok[0]), float(tok[1])), ct.body_frame(sv, float(tok[0]), float(tok[1])), phi) * ph))
    for dp in (-1e-7, 1e-7, -1e-6, 1e-6):
        print(f"period x(1{dp:+.0e})", dev(ct.brightness(A, N, eo, es, phi0 + 2 * np.pi * (jd - jd0) / (per * (1 + dp) / 24)) * ph))
r = base * ph
print("block-normalised ratio ours/theirs per block (first/last 3 points):")
for a, b in zip(edges[:-1], edges[1:]):
    q = (r[a:b] / r[a:b].mean()) / (theirs[a:b] / theirs[a:b].mean())
    print(np.round(q[:3], 5), np.round(q[-3:], 5), "trend corr with t:", np.round(np.corrcoef(jd[a:b], q)[0, 1], 3))
