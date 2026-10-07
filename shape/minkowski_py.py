"""Convex polyhedron from facet areas and outward normals (the Minkowski problem), as a fallback for DAMIT's
minkowski. convexinv's areas file can carry many (near-)zero facet areas; minkowski then never reaches its
stopping criterion (it has no iteration cap) or, with those areas raised, crashes on its fixed array
sizes. Loreley's TESS + ATLAS + survey mirror solution has 40 facets below 1e-12 of the total area and
fails in both minkowski and polyhedrec.

The body is P(h) = {x : n_i . x <= h_i}. By Brunn-Minkowski, V(h)^(1/3) is concave, so the scale-free
objective F(h) = sum(A_i h_i) / V(h)^(1/3) has a unique minimum (up to translation), where the face
areas of P(h) are proportional to A. dV/dh_i is face i's area in P(h), so
    dF/dh = A / V^(1/3) - sum(A h) a(h) / (3 V^(4/3)).
A facet with A_i ~ 0 is simply pushed out of the body (its face vanishes), which is what the
input means. Minimised with L-BFGS from the sphere h = 1; each evaluation intersects the half-spaces
(scipy HalfspaceIntersection, interior point = Chebyshev centre from a linear programme).

    V, F, info = reconstruct(areas, normals)    # F: polygons, vertex indices anticlockwise from outside
    V, F = from_areas_file(path)                 # convexinv -o output (count, then area / normal pairs)
"""
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection


def _chebyshev_centre(N, h):
    norms = np.linalg.norm(N, axis=1)
    res = linprog(np.r_[np.zeros(3), -1.0], A_ub=np.c_[N, norms], b_ub=h,
                  bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not res.success or res.x[3] <= 0:
        raise ValueError("empty polytope")
    return res.x[:3]


def polytope(N, h):
    """Vertices, hull and per-input-facet areas of P(h)."""
    c = _chebyshev_centre(N, h)
    hs = HalfspaceIntersection(np.c_[N, -h], c)
    hull = ConvexHull(hs.intersections)
    tri = hull.points[hull.simplices]
    ta = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    owner = np.argmax(hull.equations[:, :3] @ N.T, axis=1)      # input facet each hull triangle lies on
    a = np.bincount(owner, weights=ta, minlength=len(N))
    return hull, owner, a, float(hull.volume)


def reconstruct(areas, normals, tol=1e-10, maxiter=5000):
    A = np.asarray(areas, float)
    N = np.asarray(normals, float)
    N = N / np.linalg.norm(N, axis=1)[:, None]
    A = A / A.sum()

    def fg(h):
        try:
            _, _, a, V = polytope(N, h)
        except Exception:
            return np.inf, np.zeros_like(h)
        s = float(A @ h)
        return s / V ** (1 / 3), A / V ** (1 / 3) - s * a / (3 * V ** (4 / 3))

    res = minimize(fg, np.ones(len(A)), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=tol, gtol=1e-12))
    hull, owner, a, V = polytope(N, res.x)
    rel = a / a.sum()
    info = dict(converged=bool(res.success), iterations=int(res.nit), volume=V,
                max_area_error=float(np.max(np.abs(rel - A))), faces=int(len(np.unique(owner))))
    # one polygon per input facet that survives: its hull vertices ordered anticlockwise about the normal
    P, F = hull.points, []
    for i in np.unique(owner):
        idx = np.unique(hull.simplices[owner == i])
        q = P[idx] - P[idx].mean(axis=0)
        u = q[0] / np.linalg.norm(q[0])
        w = np.cross(N[i], u)
        F.append(list(idx[np.argsort(np.arctan2(q @ w, q @ u))]))
    used = np.unique(np.concatenate(F))
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    V_out = P[used]
    return V_out - V_out.mean(axis=0), [[int(i) for i in remap[f]] for f in F], info


def from_areas_file(path):
    L = open(path).read().split()
    n = int(L[0])
    x = np.array(L[1:1 + 4 * n], float).reshape(n, 4)
    V, F, info = reconstruct(x[:, 0], x[:, 1:])
    return V, F, info
