"""Drop-in, numerically-identical replacement for polyhedrec.reconstruct()
that vectorizes the O(facets^3) per-iteration residual+Jacobian evaluation
with NumPy instead of pure-Python triple-nested loops. Everything outside
that hot loop (input validation, precompute of dots/a/N, the LM root-finding
call, and the post-convergence vertex/topology extraction) is copied
verbatim from polyhedrec/polyhedrec.py so behaviour matches exactly --
only the `area()` closure body is rewritten.

Validated to reproduce polyhedrec's original output bit-for-bit (same
vertices, same facet topology) on the working test cases in
validate_fast_reconstruct.py before being trusted for larger facet counts.
"""
from __future__ import division, print_function
import warnings
import numpy as np
from scipy.optimize import root as __root

import polyhedrec as _pr
ReconstructError = _pr.ReconstructError
Polyhedron = _pr.Polyhedron
_removeDuplicates = getattr(_pr, '_polyhedrec__removeDuplicates', None) or _pr.__dict__['__removeDuplicates']


def reconstruct_fast(unormals, areas, D=None, options={}):
    options.setdefault('rtol', 1e-05)
    options.setdefault('atol', 1e-08)
    options.setdefault('ftol', 1.5e-08)
    options.setdefault('xtol', 1.5e-08)

    rtol = options['rtol']
    atol = options['atol']
    ftol = options['rtol']
    xtol = options['atol']

    unormals = [np.array(u) for u in unormals]

    if D is not None and D < 0:
        raise ValueError('D should be positive.')
    if (np.array(areas) <= 0).any():
        raise ValueError('The areas should be positive.')
    if not np.isclose(sum(a * u for a, u in zip(areas, unormals)), 0, rtol, atol).all():
        raise ValueError('The normals do not sum to zero.')
    if not np.isclose([np.dot(u, u) for u in unormals], 1).all():
        raise ValueError('The normals are not unit vectors.')

    unormals, areas = _removeDuplicates(unormals, areas, rtol, atol)
    n = len(unormals)

    dots = np.array(
        [[0] * i + [0.5] + [np.dot(unormals[i], unormals[j]) for j in range(i + 1, n)]
         for i in range(n)])
    dots = (dots + dots.T)

    a_list = []
    for k in range(n):
        spam = np.array(
            [[0] * (i + 1) + [np.dot(unormals[k], np.cross(unormals[i], unormals[j]))
             if j != k else 0 for j in range(i + 1, n)] if i != k
             else [0] * n for i in range(n)])
        spam = (spam - spam.T)
        a_list.append(spam)
    A3 = np.stack(a_list, axis=0)  # A3[k,i,j]

    dots_i = np.isclose(abs(dots), 1, rtol, atol)
    gen = [0]
    for i in range(1, n):
        if not dots_i[0, i]:
            gen.append(i)
            break
    for k in filter(lambda x: len(gen) == 2 and x not in gen, range(n)):
        if not np.isclose(a_list[k][gen[0], gen[1]], 0, rtol, atol):
            gen.append(k)
            break
    if len(gen) < 3:
        raise ValueError('The normals do not span 3D space.')

    norms = np.array([[0] * (i + 1) + [1 - dots[i, j] ** 2 for j in range(i + 1, n)]
                      for i in range(n)])
    norms = (norms + norms.T)

    N_list = [np.array([[0 if j == i
                    else -0.5 * dots[i, k] if np.isclose(norms[i, j], 0, rtol, atol)
                    else (dots[i, j] * dots[j, k] - dots[i, k]) / norms[i, j] if j != i
                    else 0 for j in range(n)] if i != k
                   else [-1] * n for i in range(n)]) for k in range(n)]
    N3 = np.stack(N_list, axis=0)  # N3[k,i,j]

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # j>i mask
    near_zero_norm = np.isclose(norms, 0, rtol, atol)

    I_idx = np.arange(n)[:, None]
    J_idx = np.arange(n)[None, :]

    # pairwise k-exclusion mask, valid_k[k,i,j] = (k!=i)&(k!=j)&upper[i,j]
    kk = np.arange(n)[:, None, None]
    valid_k = (kk != I_idx[None, :, :]) & (kk != J_idx[None, :, :]) & upper[None, :, :]

    near_zero_a = np.isclose(A3, 0, rtol, atol)

    global L, r, Lmax, Lmin
    Lmax = np.empty([n, n])
    Lmin = np.empty([n, n])

    def area(h, jac=True):
        global L, r, Lmax, Lmin

        H = np.insert(h, [gen[0], gen[1] - 1, gen[2] - 2], 0)
        r = H[None, :] - dots * H[:, None]
        np.fill_diagonal(r, 0)

        B3 = np.empty((n, n, n))
        for k in range(n):
            NK_T = N_list[k].T  # NK_T[i,j] = N[k][j,i]
            spam = (r[:, k][:, None] + r * NK_T)
            mask = upper.copy()
            mask[k, :] = False
            mask[:, k] = False
            spam = spam * mask
            B3[k] = spam + spam.T

        degenerate = near_zero_a & (B3 < 0) & valid_k
        any_degenerate = degenerate.any(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(A3 != 0, B3 / A3, 0.0)

        pos_mask = (A3 > 0) & valid_k & (~near_zero_a)
        neg_mask = (A3 < 0) & valid_k & (~near_zero_a)

        ratio_pos = np.where(pos_mask, ratio, np.inf)
        ratio_neg = np.where(neg_mask, ratio, -np.inf)

        Lmax[:, :] = np.min(ratio_pos, axis=0)
        kmax = np.argmin(ratio_pos, axis=0)
        Lmin[:, :] = np.max(ratio_neg, axis=0)
        kmin = np.argmax(ratio_neg, axis=0)

        has_pos = pos_mask.any(axis=0)
        has_neg = neg_mask.any(axis=0)

        L = np.maximum(0, Lmax - Lmin)
        L = np.where(any_degenerate, 0.0, L)
        # normals[i,j]~0 (near-parallel OR near-antiparallel pair) is a
        # separate special case in the original: it skips the whole k-loop
        # and forces L=0 directly, before any min/max tracking happens.
        L = np.where(near_zero_norm, 0.0, L)
        L = np.where(upper, L, 0.0)
        L = L + L.T

        if jac:
            c = np.zeros([n, n])
            y = [np.zeros([n, n]) for _ in range(n)]

            active = upper & (L > 0) & (~any_degenerate)
            act_has_pos = active & has_pos
            act_has_neg = active & has_neg

            ii, jj = np.nonzero(act_has_pos)
            if len(ii):
                kx = kmax[ii, jj]
                a_kij = A3[kx, ii, jj]
                N_kij = N3[kx, ii, jj]
                N_kji = N3[kx, jj, ii]
                c[ii, jj] += N_kij / a_kij
                c[jj, ii] += N_kji / a_kij
                inv_a = 1.0 / a_kij
                for kx_, i_, j_, invv in zip(kx, ii, jj, inv_a):
                    y[kx_][i_, j_] += invv
                    y[kx_][j_, i_] += invv

            ii, jj = np.nonzero(act_has_neg)
            if len(ii):
                kn = kmin[ii, jj]
                a_kij = A3[kn, ii, jj]
                N_kij = N3[kn, ii, jj]
                N_kji = N3[kn, jj, ii]
                c[ii, jj] -= N_kij / a_kij
                c[jj, ii] -= N_kji / a_kij
                inv_a = 1.0 / a_kij
                for kn_, i_, j_, invv in zip(kn, ii, jj, inv_a):
                    y[kn_][i_, j_] -= invv
                    y[kn_][j_, i_] -= invv

        A = np.array([0.5 * np.sum(np.where(np.arange(n) != i, L[i, :] * r[i, :], 0.0))
                      - areas[i] for i in range(n)])

        if jac:
            J = []
            for k in filter(lambda x: x not in gen, range(n)):
                row = []
                for i in range(n):
                    if i == k:
                        spam = np.sum(np.where(np.arange(n) != i, c[i, :] * r[i, :] - L[i, :] * dots[i, :], 0.0))
                        row.append(0.5 * spam)
                    else:
                        spam = np.sum(np.where((np.arange(n) != i) & (np.arange(n) != k), y[k][i, :] * r[i, :], 0.0))
                        if L[i, k] != 0:
                            spam += c[k, i] * r[i, k] + L[i, k]
                        row.append(0.5 * spam)
                J.append(row)
            J = np.matrix(J)
            return A, J
        else:
            return A

    coeff = np.linalg.solve([[dots[i, j] for j in gen] for i in gen], [-1] * 3)
    c0 = sum(coeff[i] * unormals[gen[i]] for i in range(3))
    h0 = np.array([1 + np.dot(unormals[i], c0)
                   for i in filter(lambda x: x not in gen, range(n))])
    if D is not None:
        d = D
    else:
        d = np.sqrt(np.average(areas / (area(h0, False) + areas)))
    h0 = d * h0

    sol = __root(area, h0, method='lm', jac=True, options={'col_deriv': 1, 'ftol': ftol, 'xtol': xtol})

    if not np.isclose(sol.fun, 0, atol, rtol).all():
        if D is None:
            raise ReconstructError(
                'The algorithm was not able to reconstruct the polyhedron; try '
                'to pass a custom value for D as input. For reference, the '
                'value computed by the algorithm was {0}.'.format(d))
        else:
            raise ReconstructError(
                'The algorithm was not able to reconstruct the polyhedron; try '
                'to pass a different value for D as input.')

    H = np.insert(sol.x, [gen[0], gen[1] - 1, gen[2] - 2], 0)

    face_ad_matrix = (L > 0).astype(int)

    num_edges = int(sum(sum(face_ad_matrix)) / 2)
    num_vertices = num_edges - n + 2
    vert_ad_matrix = np.zeros([num_vertices, num_vertices], dtype=int)

    faces = [[] for i in range(n)]
    vertices = []
    endpoints = np.full([n, n], None)

    for i in range(n):
        intersections = [j for j in range(n) if L[i, j] > 0]
        first = intersections[0]
        intersections = [intersections[k] for k in np.argsort(
            [np.arctan2(a_list[j][i, first], dots[first, j] - dots[i, j] * dots[first, i])
             for j in intersections])]
        for k in range(len(intersections)):
            j = intersections[k]
            nextj = intersections[(k + 1) % len(intersections)]
            if j > i:
                if nextj > i:
                    o_ij = (r[j, i] * unormals[i] + r[i, j] * unormals[j]) / norms[i, j]
                    vertex = o_ij + Lmax[i, j] * np.cross(unormals[i], unormals[j])
                    vertices.append(vertex)
                    idx = len(vertices) - 1
                else:
                    idx = endpoints[nextj, i]
            else:
                idx = endpoints[i, j]
            endpoints[i, j] = idx
            endpoints[nextj, i] = idx
            faces[i].append(idx)
        for j in intersections:
            vert_ad_matrix[endpoints[i, j], endpoints[j, i]] = 1

    del L, Lmin, Lmax, r
    return Polyhedron(vertices, vert_ad_matrix, faces, face_ad_matrix, unormals, areas, H)
