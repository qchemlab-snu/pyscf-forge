#!/usr/bin/env python
#
# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jiseong Park <fark4308@snu.ac.kr>
#          Minseok Oh <msjeff2001@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>


"""Frozen Active-Space SCF tools built on top of PySCF.

This module keeps selected active orbitals fixed while reoptimizing the
complementary orbital space with SCF-like iterations. It is based on
docs/ref_fasscf.py, with a class-centered API and a small helper surface.

EDIIS + DIIS bath solver (ediis_kernel, ediis_kernel_many; the first stage
of mixed_routine).  With the active orbitals and the spin-summed active
density A frozen, the bath energy is a closed-shell functional of the core
projector P = u u^T in the inactive orbital basis:

    E(P) = 2 tr(P h') + tr(P g[P]),   F(P) = h' + g[P],
    h' = h + (J - K/2)[A],   g[P] = (2J - K)[P]   (linear in P).

Only the spin-summed A enters (the inactive block of the ROHF Fock is
(F_a + F_b)/2), so this is the functional of kernel() and
newton_soscf_kernel().  E is exactly quadratic in P, so the EDIIS energy of
Kudin, Scuseria and Cances (J. Chem. Phys. 116, 8255 (2002), Eq. (8)) is
exact here also for the interpolated P = sum_i c_i P_i:

    E(sum_i c_i P_i) = sum_i c_i E_i
                       - 1/2 sum_ij c_i c_j tr((F_i - F_j)(P_i - P_j)).

While max |F P - P F| > ediis_switch, EDIIS minimizes this energy on the
simplex c_i >= 0, sum c_i = 1 (the global minimum over all faces, their
Eq. (16)); below it Pulay's commutator DIIS is used, with a history that
starts at the switch.  DIIS goes back to EDIIS when the error rises above
ediis_switch and above 10 times its smallest DIIS value.  After
convergence a negative lowest orbital-Hessian eigenvalue (Davidson) marks a
saddle point: a step along its eigenvector and a new EDIIS stage follow
(max_stability).  All baths of one GBCI run are solved in lockstep with one
batched get_jk call per step.
"""

import itertools
from contextlib import contextmanager
from types import MethodType

import numpy as np
import scipy.linalg

from pyscf import __config__, lib
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf.soscf import ciah, newton_ah
from pyscf.gbci.gbci import str2occ


TIGHT_GRAD_CONV_TOL = getattr(
    __config__, "scf_hf_kernel_tight_grad_conv_tol", True
)
MISSING = object()
FOCK_ERROR = (
    "FASSCF inactive-block diagonalization requires a 2D effective Fock matrix. "
    "Use RHF/ROHF-like PySCF objects or provide an effective 2D Fock."
)


def _inactive_orbital_gradient(mf, mo_coeff, mo_occ, fock, active_idx):
    """Project the ROHF orbital gradient onto FASSCF rotations."""
    g_full = np.asarray(mf.get_grad(mo_coeff, mo_occ, fock))

    active_mask = np.zeros(mo_occ.size, dtype=bool)
    active_mask[active_idx] = True
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:, None] & occidxa[None, :]
    uniq_var_b = viridxb[:, None] & occidxb[None, :]
    uniq_ab = uniq_var_a | uniq_var_b
    inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
    inactive_var = inactive_rotation[uniq_ab]

    if inactive_var.size != g_full.size:
        raise RuntimeError(
            "Unable to map FASSCF orbital gradient to inactive-only space."
        )
    return g_full[inactive_var]


class FASSCFResult:
    """Container returned by FASSCF kernels."""

    def __init__(
        self,
        converged,
        e_tot,
        mo_energy,
        mo_coeff,
        mo_occ=None,
        dm=None,
        cycles=0,
        mode="fasscf",
    ):
        self.converged = converged
        self.e_tot = e_tot
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.dm = dm
        self.cycles = cycles
        self.mode = mode

    def as_tuple(self):
        """Return a tuple close to the original reference-code API."""
        if self.mode in (
            "group_average",
            "group_average_soscf",
            "group_average_newton_soscf",
            "group_average_ediis",
        ):
            return self.converged, self.e_tot, self.mo_energy, self.mo_coeff
        return self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ


def _group_average_occ(ncas, nelecas, group_info_list, target_group):
    """Alpha and beta active occupations averaged over the determinants of
    target_group (equal weights, as GroupAverageFASSCF.kernel)."""
    occs = []
    for n in (int(nelecas[0]), int(nelecas[1])):
        strs = np.asarray(cistring.make_strings(range(ncas), n))
        occs.append(((strs[:, None] >> np.arange(ncas)) & 1).astype(float))
    nb = occs[1].shape[0]
    flat = np.asarray(group_info_list).reshape(-1)
    if flat.size != occs[0].shape[0] * nb:
        raise ValueError(
            "group_info_list size does not match the alpha/beta determinant space."
        )
    conf = np.nonzero(flat == target_group)[0]
    if conf.size == 0:
        raise ValueError(f"No active-space configurations found for {target_group!r}.")
    return occs[0][conf // nb].mean(axis=0), occs[1][conf % nb].mean(axis=0)


def _complement(u):
    """Orthonormal complement of the orthonormal columns u."""
    nin, nocc = u.shape
    _, v = np.linalg.eigh(np.eye(nin) - lib.dot(u, u.T))
    return v[:, nocc:]


class _State:
    """Core orbitals u, virtual orbitals uv, energy e and Fock matrix f of one
    bath point (inactive orbital basis)."""

    def __init__(self, u, e, f, uv=None):
        self.u = u
        self.uv = _complement(u) if uv is None else uv
        self.e = e
        self.f = f
        fu = lib.dot(f, u)
        self.foo = lib.dot(u.T, fu)
        self.fvo = lib.dot(self.uv.T, fu)
        self.fvv = lib.dot(self.uv.T, lib.dot(f, self.uv))
        # 2 ||F_vo||, the norm of the inactive ROHF orbital gradient
        self.norm_g = 2 * np.linalg.norm(self.fvo)


class _EDIIS:
    """EDIIS + DIIS history of one bath.

    Entries: projector P_i, Fock matrix F_i, energy E_i and commutator error
    e_i = F_i P_i - P_i F_i.  update() adds a point and returns the Fock
    matrix to diagonalize.  The traces T_ij = tr(F_i P_j) and the overlaps
    <e_i, e_j> are extended by one row and column per entry."""

    max_ediis = 10       # EDIIS uses at most this many newest entries
    back_factor = 10.
    _faces = {}          # n -> index arrays of the faces of the simplex

    def __init__(self, space, switch):
        self.space = space
        self.switch = switch
        self.p, self.f, self.e, self.err = [], [], [], []
        self._t = np.zeros((0, 0))
        self._o = np.zeros((0, 0))
        self.mode = None
        self.err_min = None
        self._bar = None     # (P, F, E) of the last EDIIS interpolation

    def _keep(self, idx):
        idx = list(idx)
        for name in ("p", "f", "e", "err"):
            lst = getattr(self, name)
            setattr(self, name, [lst[i] for i in idx])
        self._t = self._t[np.ix_(idx, idx)]
        self._o = self._o[np.ix_(idx, idx)]

    def _push(self, p, f, e):
        if len(self.p) == self.space:
            self._keep(range(1, len(self.p)))
        err = lib.dot(f, p)
        err = err - err.T
        n = len(self.p)
        t = np.empty((n + 1, n + 1))
        o = np.empty((n + 1, n + 1))
        t[:n, :n] = self._t
        o[:n, :n] = self._o
        for k in range(n):
            t[n, k] = np.vdot(f, self.p[k])
            t[k, n] = np.vdot(self.f[k], p)
            o[n, k] = o[k, n] = np.vdot(err, self.err[k])
        t[n, n] = np.vdot(f, p)
        o[n, n] = np.vdot(err, err)
        self._t, self._o = t, o
        self.p.append(p)
        self.f.append(f)
        self.e.append(e)
        self.err.append(err)

    def bmat(self, idx):
        """B_ij = tr((F_i - F_j)(P_i - P_j)) of the entries idx."""
        t = self._t[np.ix_(idx, idx)]
        d = np.diag(t)
        return d[:, None] + d[None, :] - t - t.T

    def model(self, c, idx):
        """EDIIS energy sum_i c_i E_i - 1/2 c^T B c."""
        c = np.asarray(c, float)
        return float(c @ np.asarray(self.e)[idx] - .5 * c @ self.bmat(idx) @ c)

    @classmethod
    def _face_sets(cls, n):
        if n not in cls._faces:
            cls._faces[n] = [np.array(list(itertools.combinations(range(n), m)),
                                      dtype=int) for m in range(1, n + 1)]
        return cls._faces[n]

    @classmethod
    def solve_simplex(cls, e, b):
        """Global minimum of e.c - 1/2 c^T b c on c >= 0, sum c = 1, from the
        stationary points of all faces (the faces of one size together)."""
        e = np.asarray(e, float)
        n = e.size
        e = e - e.min()
        best_f, best_c = np.inf, None
        for s in cls._face_sets(n):
            nf, m = s.shape
            a = np.zeros((nf, m + 1, m + 1))
            a[:, :m, :m] = -b[s[:, :, None], s[:, None, :]]
            a[:, :m, m] = a[:, m, :m] = 1.
            rhs = np.concatenate((-e[s], np.ones((nf, 1))), axis=1)
            try:
                x = np.linalg.solve(a, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                x = np.full(rhs.shape, np.nan)
                for r in range(nf):
                    try:
                        x[r] = np.linalg.solve(a[r], rhs[r])
                    except np.linalg.LinAlgError:
                        pass
            cs = x[:, :m]
            ok = np.all(np.isfinite(cs), axis=1) & (cs.min(axis=1) >= -1e-12)
            if not ok.any():
                continue
            cs = np.maximum(cs[ok], 0.)
            cs /= cs.sum(axis=1, keepdims=True)
            c = np.zeros((cs.shape[0], n))
            c[np.arange(cs.shape[0])[:, None], s[ok]] = cs
            f = c @ e - .5 * np.einsum("ki,ij,kj->k", c, b, c)
            k = int(np.argmin(f))
            if f[k] < best_f:
                best_f, best_c = f[k], c[k]
        return best_c

    def _ediis(self):
        n = len(self.p)
        idx = list(range(max(0, n - self.max_ediis), n))
        c = self.solve_simplex(np.asarray(self.e)[idx], self.bmat(idx))
        if c[-1] < 1e-10 and self._bar is not None:
            # no weight on the new point (history truncated): put the last
            # interpolated point back, the new point lowers the energy from it
            self._push(*self._bar)
            n = len(self.p)
            self._keep(list(range(n - 2)) + [n - 1, n - 2])
            idx = list(range(max(0, n - self.max_ediis), n))
            c = self.solve_simplex(np.asarray(self.e)[idx], self.bmat(idx))
        p = sum(ci * self.p[i] for ci, i in zip(c, idx))
        f = sum(ci * self.f[i] for ci, i in zip(c, idx))
        self._bar = (p, f, self.model(c, idx))
        return f

    def _cdiis(self):
        n = len(self.err)
        h = np.zeros((n + 1, n + 1))
        h[1:, 1:] = self._o
        h[0, 1:] = h[1:, 0] = 1.
        g = np.zeros(n + 1)
        g[0] = 1.
        w, v = scipy.linalg.eigh(h)
        keep = abs(w) > 1e-14
        c = (v[:, keep] @ ((v[:, keep].T @ g) / w[keep]))[1:]
        return sum(ci * fi for ci, fi in zip(c, self.f))

    def update(self, st, p):
        self._push(p, st.f, st.e)
        err = abs(self.err[-1]).max()
        if self.mode == "cdiis":
            self.err_min = min(self.err_min, err)
            use_ediis = err > max(self.switch, self.back_factor * self.err_min)
        else:
            use_ediis = err > self.switch
            self.err_min = err
        if use_ediis:
            self.mode = "ediis"
            return self._ediis()
        if self.mode == "ediis":
            # EDIIS points are interpolations: DIIS starts from the newest
            self._keep([len(self.p) - 1])
        self.mode = "cdiis"
        self._bar = None
        return self._cdiis()


class _Bath:
    def __init__(self, hprime, label):
        self.hprime = hprime
        self.label = label
        self.st = None
        self.conv = False
        self.cycles = 0
        self.stage = "ediis"
        self.hess_min = None
        self.u_imom = None


class _BathEDIIS:
    """Lockstep EDIIS + DIIS and stability following of baths that share
    the reference orbitals.

    orth : inactive reference orbitals (nao, nin), the basis of u
    hprimes : h + (J - K/2)[A] of every bath (AO)
    u0 : start core orbitals (nin, ncore)
    Rotations x (virtual x core), C <- C expm(K) with K[v,o] = x.  Gradient
    4 F_vo; Hessian 4 [F_vv x - x F_oo + C_v^T (2J - K)[C_v x C_o^T + h.c.] C_o].
    """

    def __init__(self, mf, orth, ncore, hprimes, labels, log, conv_tol,
                 conv_tol_grad, max_cycle, switch, space, occupation,
                 max_stability, stability_tol, accept_grad=None):
        if occupation not in ("aufbau", "mom", "imom"):
            raise ValueError(
                "ediis_occupation must be 'aufbau', 'mom' or 'imom', not %r"
                % occupation)
        self.mf = mf
        self.orth = orth
        self.ncore = ncore
        self.baths = [_Bath(h, lab) for h, lab in zip(hprimes, labels)]
        self.log = log
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        # gradient below which a bath that stops at max_cycle is converged
        self.accept_grad = conv_tol_grad if accept_grad is None else accept_grad
        self.max_cycle = max_cycle
        self.switch = switch
        self.space = space
        self.occupation = occupation
        self.max_stability = max_stability
        self.stability_tol = stability_tol

    def veff(self, dms):
        """(J - K/2)[dm] for a stack of symmetric densities."""
        vj, vk = self.mf.get_jk(self.mf.mol, dms, hermi=1)
        return np.asarray(vj - .5 * vk).reshape(np.shape(dms))

    def states(self, ks, us, uvs=None):
        if not ks:
            return []
        dms = np.asarray([2 * lib.dot(c, c.T) for c in
                          (lib.dot(self.orth, u) for u in us)])
        if self.ncore:
            vcs = self.veff(dms)
        else:
            vcs = np.zeros_like(dms)
        out = []
        for i, (k, u, dm, vc) in enumerate(zip(ks, us, dms, vcs)):
            hp = self.baths[k].hprime
            e = float(np.einsum("pq,qp", dm, hp + .5 * vc))
            f = lib.dot(self.orth.T, lib.dot(hp + vc, self.orth))
            out.append(_State(u, e, f, None if uvs is None else uvs[i]))
        return out

    @staticmethod
    def rotate(st, x):
        nin, ncore = st.u.shape
        k = np.zeros((nin, nin))
        k[ncore:, :ncore] = x.reshape(st.fvo.shape)
        k[:ncore, ncore:] = -k[ncore:, :ncore].T
        c = lib.dot(np.hstack((st.u, st.uv)), scipy.linalg.expm(k))
        return c[:, :ncore], c[:, ncore:]

    def diis_many(self, idx, switch):
        """Lockstep EDIIS + DIIS of the baths idx from bath.st.  A bath that
        does not converge keeps its iterate of smallest gradient.  Returns
        the converged flags and the cycles (dicts over idx)."""
        baths, log = self.baths, self.log
        dii = {k: _EDIIS(self.space, switch) for k in idx}
        best = {k: baths[k].st for k in idx}
        e_last = {k: None for k in idx}
        conv = {k: False for k in idx}
        cyc = {k: 0 for k in idx}
        active = list(idx)
        for cycle in range(1, self.max_cycle + 1):
            prop, prop_v, nxt = [], [], []
            for k in active:
                b = baths[k]
                st = b.st
                cyc[k] = cycle
                if st.norm_g < best[k].norm_g:
                    best[k] = st
                de = np.inf if e_last[k] is None else st.e - e_last[k]
                if abs(de) < self.conv_tol and st.norm_g < self.conv_tol_grad:
                    log.debug1("bath %s EDIIS cycle= %d E_core= %.15g "
                               "delta_E= %4.3g |g_bath|= %4.3g converged",
                               b.label, cycle, st.e, de, st.norm_g)
                    conv[k] = True
                    continue
                p = lib.dot(st.u, st.u.T)
                f = dii[k].update(st, p)
                log.debug1("bath %s EDIIS cycle= %d E_core= %.15g "
                           "delta_E= %4.3g |g_bath|= %4.3g next: %s",
                           b.label, cycle, st.e, de, st.norm_g, dii[k].mode)
                w, y = scipy.linalg.eigh(f)
                if self.occupation == "aufbau":
                    occ = np.arange(self.ncore)
                else:
                    ref = (b.u_imom if self.occupation == "imom"
                           and b.u_imom is not None else st.u)
                    proj = np.einsum("ij,ij->j", y, lib.dot(ref, lib.dot(ref.T, y)))
                    occ = np.sort(np.argsort(-proj, kind="stable")[:self.ncore])
                vir = np.setdiff1d(np.arange(y.shape[1]), occ)
                e_last[k] = st.e
                prop.append(y[:, occ])
                prop_v.append(y[:, vir])
                nxt.append(k)
            active = nxt
            if not active:
                break
            for k, st in zip(active, self.states(active, prop, prop_v)):
                baths[k].st = st
        for k in idx:
            if not conv[k] and best[k].norm_g < baths[k].st.norm_g:
                baths[k].st = best[k]
            if not conv[k] and baths[k].st.norm_g < self.accept_grad:
                conv[k] = True
        return conv, cyc

    def hx_many(self, ks, xs):
        """Orbital Hessian times x for several baths, one batched J/K."""
        d1s = []
        cos = []
        for k, x in zip(ks, xs):
            st = self.baths[k].st
            co = lib.dot(self.orth, st.u)
            a = lib.dot(self.orth, lib.dot(st.uv, x.reshape(st.fvo.shape)))
            d1 = lib.dot(a, co.T)
            d1s.append(d1 + d1.T)
            cos.append(co)
        v1s = self.veff(np.asarray(d1s))
        out = []
        for k, x, v1, co in zip(ks, xs, v1s, cos):
            st = self.baths[k].st
            x = x.reshape(st.fvo.shape)
            cv = lib.dot(self.orth, st.uv)
            hx = (lib.dot(st.fvv, x) - lib.dot(x, st.foo)
                  + 2 * lib.dot(cv.T, lib.dot(v1, co)))
            out.append((4 * hx).ravel())
        return out

    def lowest_many(self, ks, x0, tol=1e-3, nstart=3, max_cycle=60):
        """Lowest orbital-Hessian eigenvalue of the baths ks (lockstep
        Davidson).  x0: {k: previous eigenvector as a generator in the
        inactive basis}; else the unit vectors of the nstart lowest diagonal
        elements.  Returns ({k: eigenvalue}, {k: eigenvector})."""
        baths = self.baths
        hd, vs, avs, lam, vec = {}, {}, {}, {}, {}
        k0s, x0s = [], []
        for k in ks:
            st = baths[k].st
            hd[k] = 4 * (np.diag(st.fvv)[:, None] - np.diag(st.foo)[None, :]).ravel()
            xs = []
            if k in x0:
                x = lib.dot(st.uv.T, lib.dot(x0[k], st.u)).ravel()
                nx = np.linalg.norm(x)
                if nx > 1e-3:
                    xs = [x / nx]
            if not xs:
                for i in np.argsort(hd[k])[:nstart]:
                    x = np.zeros_like(hd[k])
                    x[i] = 1.
                    xs.append(x)
            vs[k], avs[k] = xs, []
            k0s += [k] * len(xs)
            x0s += xs
        for k, ax in zip(k0s, self.hx_many(k0s, x0s)):
            avs[k].append(ax)
        active = list(ks)
        for _ in range(max_cycle):
            new_k, new_t = [], []
            for k in active:
                v = np.array(vs[k])
                av = np.array(avs[k])
                h = lib.dot(v, av.T)
                w, y = np.linalg.eigh(.5 * (h + h.T))
                x = y[:, 0] @ v
                r = y[:, 0] @ av - w[0] * x
                lam[k], vec[k] = float(w[0]), x
                if np.linalg.norm(r) < tol:
                    continue
                d = hd[k] - w[0]
                d[abs(d) < 1e-4] = 1e-4
                t = r / d
                for _ in range(2):
                    t -= v.T @ (v @ t)
                nt = np.linalg.norm(t)
                if nt < 1e-8 or len(vs[k]) >= 40:
                    continue
                new_k.append(k)
                new_t.append(t / nt)
            if not new_k:
                break
            for k, t, at in zip(new_k, new_t, self.hx_many(new_k, new_t)):
                vs[k].append(t)
                avs[k].append(at)
            active = new_k
        return lam, vec

    def follow(self, idx):
        """Stability following of the converged baths idx: a lowest Hessian
        eigenvalue below -stability_tol gives a step along its eigenvector
        (best of 8 trial points) and a new EDIIS stage (switch 1e-3)."""
        baths, log = self.baths, self.log
        check = [k for k in idx if baths[k].conv and baths[k].st.fvo.size]
        x0 = {}
        for _ in range(self.max_stability):
            if not check:
                break
            lam, vec = self.lowest_many(check, x0)
            saddle = []
            for k in check:
                baths[k].hess_min = lam[k]
                if lam[k] < -self.stability_tol:
                    saddle.append(k)
            if not saddle:
                break
            trial_k, trial_u, trial_v = [], [], []
            for k in saddle:
                for a in (.1, .2, .4, .8):
                    for sgn in (1, -1):
                        u, uv = self.rotate(baths[k].st, sgn * a * vec[k])
                        trial_k.append(k)
                        trial_u.append(u)
                        trial_v.append(uv)
            trials = self.states(trial_k, trial_u, trial_v)
            moved = []
            for k in saddle:
                b = baths[k]
                best = min((t for kk, t in zip(trial_k, trials) if kk == k),
                           key=lambda t: t.e)
                log.note("bath %s: saddle point (lowest Hessian eigenvalue "
                         "%.3g, E_core = %.10f); stability step to %.10f",
                         b.label, lam[k], b.st.e, best.e)
                if best.e < b.st.e:
                    st = b.st
                    x0[k] = lib.dot(st.uv, lib.dot(vec[k].reshape(st.fvo.shape),
                                                   st.u.T))
                    b.st = best
                    b.u_imom = best.u
                    moved.append(k)
            if not moved:
                break
            conv, cyc = self.diis_many(moved, 1e-3)
            for k in moved:
                b = baths[k]
                b.cycles += cyc[k]
                b.stage += "+stability"
                b.conv = conv[k]
            check = [k for k in moved if baths[k].conv]

    def run(self, u0):
        n = len(self.baths)
        for b, st in zip(self.baths, self.states(list(range(n)), [u0] * n)):
            b.st = st
            b.u_imom = u0
        if self.ncore == 0:
            for b in self.baths:
                b.conv, b.stage = True, "trivial"
            return self.baths
        idx = list(range(n))
        conv, cyc = self.diis_many(idx, self.switch)
        for k in idx:
            self.baths[k].conv, self.baths[k].cycles = conv[k], cyc[k]
        if self.max_stability:
            self.follow(idx)
        return self.baths

class FASSCF(rohf.ROHF):
    """ROHF-derived optimizer with frozen active orbitals.

    Attributes of the EDIIS + DIIS solver (ediis_kernel):
        ediis_switch : float
            EDIIS -> DIIS switch on max |F P - P F| (default 0.1).
        ediis_occupation : str
            Core orbitals of each step: 'aufbau' (default, lowest orbital
            energies, as kernel()), 'mom' (maximum overlap with the previous
            core) or 'imom' (maximum overlap with the start core).
        max_stability : int
            Steps of the stability following (default 5, 0 = off).
        stability_tol : float
            A lowest orbital-Hessian eigenvalue below -stability_tol marks a
            saddle point (default 1e-4).
        diis_space : int
            History length of EDIIS/DIIS (SCF attribute, default 8).
        conv_tol_grad : float
            With the default None the EDIIS solver converges the bath
            gradient to 1e-3 sqrt(conv_tol), not below 1e-8 (DIIS converges
            fast near the solution); a bath that reaches max_cycle
            with a gradient below sqrt(conv_tol) (the threshold of kernel())
            counts as converged.
    """

    ediis_switch = getattr(__config__, "gbci_FASSCF_ediis_switch", 0.1)
    ediis_occupation = getattr(__config__, "gbci_FASSCF_ediis_occupation", "aufbau")
    max_stability = getattr(__config__, "gbci_FASSCF_max_stability", 5)
    stability_tol = getattr(__config__, "gbci_FASSCF_stability_tol", 1e-4)
    _keys = {"ediis_switch", "ediis_occupation", "max_stability", "stability_tol"}

    def __init__(
        self,
        gbci,
        target=None,
        ncas=None,
        ncore=None,
        nelecas=None,
        mo_coeff=None,
        mo_energy=None,
        conv_tol=1e-10,
        conv_tol_grad=None,
        max_cycle=100,
        dump_chk=True,
        conv_check=True,
        callback=None,
        scf_options=None,
        restore_on_failure=True,
    ):
        mf = gbci._scf

        if hasattr(mf, "mol"):
            rohf.ROHF.__init__(self, mf.mol)
            for key, value in mf.__dict__.items():
                if key not in ("_chkfile", "chkfile"):
                    setattr(self, key, value)
        else:
            rohf.ROHF.__init__(self, mf)

        ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
        ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
        mo_coeff = getattr(gbci, "mo_coeff", None) if mo_coeff is None else mo_coeff
        mo_energy = getattr(gbci, "mo_energy", None) if mo_energy is None else mo_energy

        active_orbitals = range(ncore, ncore+ ncas)
        core_orbitals = range(ncore)

        self.gbci = gbci
        self.target = target
        self.ncas = ncas
        self.ncore = ncore
        self.nelecas = nelecas
        self.active_orbitals = None if active_orbitals is None else list(active_orbitals)
        self.core_orbitals = None if core_orbitals is None else list(core_orbitals)
        if mo_coeff is not None:
            self.mo_coeff = mo_coeff
        if mo_energy is not None:
            self.mo_energy = mo_energy
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.max_cycle = max_cycle
        self.fasscf_dump_chk = dump_chk
        self.conv_check = conv_check
        self.callback = callback
        self.scf_options = {"diis": False}
        self.scf_options.update(scf_options or {})
        self.restore_on_failure = restore_on_failure

        self.converged = False
        self.e_tot = None
        self.mo_occ = None
        self.dm = None
        self.cycles = 0
        self.last_result = None

    def set_scf_options(self, **options):
        """Update default SCF controls used by subsequent kernels."""
        self.scf_options.update(options)
        return self

    def active_occ_from_target(self, target=None):
        """Return the active occupation selected by target."""
        if target is None:
            target = self.target
        if target is None:
            raise ValueError("target is required when active_occ is not provided.")

        target_arr = np.asarray(target)
        if target_arr.ndim != 0:
            return np.asarray(target_arr, dtype=float)

        po_list = self.gbci.possible_occ()
        target_index = int(target_arr)
        if target_index < 0 or target_index >= len(po_list):
            raise IndexError(
                f"target index {target_index} is outside possible_occ size {len(po_list)}."
            )
        return np.asarray(po_list[target_index], dtype=float)

    def kernel(
        self,
        target=None,
        mo_coeff=None,
        mo_energy=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        conv_check=None,
        scf_options=None,
        **scf_kwargs,
    ):
        """Run active-frozen FASSCF."""
        if "init_dm" in scf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        mol = mf.mol
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
            logger.info(mf, "Set gradient conv threshold to %g", conv_tol_grad)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        conv_check = self.conv_check if conv_check is None else conv_check
        callback = self.callback if callback is None else callback

        mo_coeff, mo_energy, active_idx, core_idx, active_occ, mo_occ = (
            self.prepare_orbitals(
                mo_coeff, mo_energy, active_orbitals, core_orbitals, active_occ,
                target,
            )
        )
        initial_mo_coeff = np.array(mo_coeff, copy=True)
        options = self.merged_options(
            scf_options,
            scf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )

        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(mf, options):
            dm = mf.make_rdm1(mo_coeff, mo_occ) if dm0 is None else dm0
            h1e = mf.get_hcore(mol)
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "init E= %.15g", e_tot)

            s1e = mf.get_ovlp(mol)
            self.log_overlap_condition(mf, s1e, conv_tol)

            scf_conv = False
            cycles = 0
            if max_cycle <= 0:
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)
                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                return self.finish(
                    FASSCFResult(False, e_tot, mo_energy, mo_coeff, mo_occ, dm, 0)
                )

            mf_diis = self.make_diis(mf, h1e, s1e, vhf, dm)
            self.call_hook(mf, "pre_kernel", locals())
            cput1 = logger.timer(mf, "initialize scf", *cput0)
            fock_last = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = np.asarray(
                    mf.get_fock(
                        h1e, s1e, vhf, dm,
                        cycle=cycle, diis=mf_diis, fock_last=fock_last,
                    )
                )
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)

                fock_last = fock
                fock_plain = mf.get_fock(h1e, s1e, vhf, dm)
                if np.asarray(fock_plain).ndim != 2:
                    raise ValueError(FOCK_ERROR)
                g_inact = _inactive_orbital_gradient(
                    mf, mo_coeff, mo_occ, fock_plain, active_idx)
                norm_gorb = np.linalg.norm(g_inact)
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                logger.info(
                    mf,
                    "cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                elif (abs(e_tot - last_hf_e) < conv_tol and
                      norm_gorb < conv_tol_grad):
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())

                cput1 = logger.timer(mf, f"cycle= {cycles}", *cput1)
                if scf_conv:
                    break

            if scf_conv and conv_check:
                dm_last = dm
                last_hf_e = e_tot
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                fock = mf.get_fock(h1e, s1e, vhf, dm)
                if np.asarray(fock).ndim != 2:
                    raise ValueError(FOCK_ERROR)
                g_inact = _inactive_orbital_gradient(
                    mf, mo_coeff, mo_occ, fock, active_idx)
                norm_gorb = np.linalg.norm(g_inact)
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                loose_e_tol = conv_tol * 10
                loose_g_tol = conv_tol_grad * 3
                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                else:
                    scf_conv = abs(e_tot - last_hf_e) < loose_e_tol or norm_gorb < loose_g_tol
                logger.info(
                    mf,
                    "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )
                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())

            if getattr(mf, "disp", None) is not None:
                e_disp = mf.get_dispersion()
                mf.scf_summary["dispersion"] = e_disp
                e_tot += e_disp

            logger.timer(mf, "scf_cycle", *cput0)
            self.call_hook(mf, "post_kernel", locals())

        if not scf_conv and self.restore_on_failure:
            mo_coeff = initial_mo_coeff

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="fasscf",
            )
        )

    run = kernel

    def soscf_kernel(
        self,
        target=None,
        mo_coeff=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dm0=None,
        freeze_active=True,
        canonicalization=False,
        dump_chk=None,
        callback=None,
        scf_options=None,
        **soscf_kwargs,
    ):
        """Run inactive-only SOSCF with the active orbitals frozen."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        mo_coeff, _, active_idx, _, _, mo_occ = self.prepare_orbitals(
            mo_coeff, None, active_orbitals, core_orbitals, active_occ, target
        )
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        soscf = mf.newton()
        options = self.merged_options(
            scf_options,
            soscf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(soscf, options):
            h1e = mf.get_hcore(mol)
            s1e = mf.get_ovlp(mol)
            dm = mf.make_rdm1(mo_coeff, mo_occ) if dm0 is None else dm0
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "SOSCF init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                g_full, h_op_full, h_diag_full = soscf.gen_g_hop(
                    mo_coeff, mo_occ, fock
                )

                active_mask = np.zeros(mo_occ.size, dtype=bool)
                active_mask[active_idx] = True
                occidxa = mo_occ > 0
                occidxb = mo_occ == 2
                viridxa = ~occidxa
                viridxb = ~occidxb
                uniq_var_a = viridxa[:, None] & occidxa[None, :]
                uniq_var_b = viridxb[:, None] & occidxb[None, :]
                uniq_ab = uniq_var_a | uniq_var_b
                if freeze_active:
                    inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
                else:
                    inactive_rotation = np.ones((mo_occ.size, mo_occ.size), dtype=bool)
                inactive_var = inactive_rotation[uniq_ab]

                g_full = np.asarray(g_full)
                if inactive_var.size != g_full.size:
                    raise RuntimeError(
                        "Unable to map ROHF SOSCF variables to inactive-only space."
                    )
                g_orb = g_full[inactive_var]
                h_diag = np.asarray(h_diag_full)[inactive_var]
                norm_gorb = np.linalg.norm(g_orb)

                if g_orb.size == 0:
                    scf_conv = True
                    break

                def h_op(x):
                    x_full = np.zeros_like(g_full)
                    x_full[inactive_var] = x
                    return np.asarray(h_op_full(x_full))[inactive_var]

                def g_op():
                    return g_orb

                def precond(x, e):
                    hdiagd = h_diag - (e - soscf.ah_level_shift)
                    hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                    return x / hdiagd

                dx = np.zeros_like(g_orb)
                ah_conv_tol = min(norm_gorb**2, soscf.ah_conv_tol)
                ah_start_tol = min(norm_gorb * 5, soscf.ah_start_tol)
                for ah_end, ihop, w, dxi, hdxi, residual, seig in ciah.davidson_cc(
                    h_op,
                    g_op,
                    precond,
                    g_orb,
                    tol=ah_conv_tol,
                    max_cycle=soscf.ah_max_cycle,
                    lindep=soscf.ah_lindep,
                    verbose=logger.new_logger(soscf, soscf.verbose),
                ):
                    if (
                        ah_end
                        or ihop == soscf.ah_max_cycle
                        or (
                            np.linalg.norm(residual) < ah_start_tol
                            and ihop >= soscf.ah_start_cycle
                        )
                        or seig < soscf.ah_lindep
                    ):
                        dx = np.array(dxi, copy=True)
                        dxmax = np.max(abs(dx)) if dx.size else 0
                        if dxmax > soscf.max_stepsize:
                            dx *= soscf.max_stepsize / dxmax
                        break

                dx_full = np.zeros_like(g_full)
                dx_full[inactive_var] = dx
                u = soscf.update_rotate_matrix(dx_full, mo_occ, mo_coeff=mo_coeff)
                mo_coeff = soscf.rotate_mo(mo_coeff, u)
                if freeze_active:
                    mo_coeff[:, active_idx] = active_mo_coeff

                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                fock_mo = mo_coeff.conj().T @ np.asarray(fock) @ mo_coeff
                mo_energy = np.diag(fock_mo)
                if np.allclose(mo_energy.imag, 0):
                    mo_energy = mo_energy.real
                e_tot = mf.energy_tot(dm, h1e, vhf)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                logger.info(
                    mf,
                    "SOSCF cycle= %d E= %.15g  delta_E= %4.3g  |g_bath|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                if scf_conv:
                    break

            logger.timer(mf, "inactive-only SOSCF", *cput0)

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="soscf",
            )
        )

    soscf = soscf_kernel

    def newton_soscf_kernel(
        self,
        target=None,
        mo_coeff=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dm0=None,
        freeze_active=True,
        canonicalization=False,
        dump_chk=None,
        callback=None,
        scf_options=None,
        **newton_kwargs,
    ):
        """Run PySCF Newton SOSCF with active-involving rotations removed."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        mo_coeff, _, active_idx, _, _, mo_occ = self.prepare_orbitals(
            mo_coeff, None, active_orbitals, core_orbitals, active_occ, target
        )
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        soscf = mf.newton()
        original_gen_g_hop = soscf.gen_g_hop
        original_get_grad = soscf.get_grad
        original_update_rotate_matrix = soscf.update_rotate_matrix
        original_rotate_mo = soscf.rotate_mo

        def gen_g_hop_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None, *args, **kwargs):
            g_full, h_op_full, h_diag_full = original_gen_g_hop(
                mo_coeff1, mo_occ1, fock_ao, *args, **kwargs
            )
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]

            g_full = np.asarray(g_full)
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map Newton SOSCF variables to inactive-only space."
                )
            g_orb = g_full[inactive_var]
            h_diag = np.asarray(h_diag_full)[inactive_var]

            def h_op(x):
                x_full = np.zeros_like(g_full)
                x_full[inactive_var] = x
                return np.asarray(h_op_full(x_full))[inactive_var]

            return g_orb, h_op, h_diag

        def get_grad_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None):
            g_full = np.asarray(original_get_grad(mo_coeff1, mo_occ1, fock_ao))
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map Newton SOSCF gradient to inactive-only space."
                )
            return g_full[inactive_var]

        def update_rotate_matrix_inactive(
            newton_mf, dx, mo_occ1, u0=1, mo_coeff=None
        ):
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if np.asarray(dx).size == int(np.count_nonzero(inactive_var)):
                dx_full = np.zeros(inactive_var.size, dtype=np.asarray(dx).dtype)
                dx_full[inactive_var] = dx
            else:
                dx_full = dx
            return original_update_rotate_matrix(dx_full, mo_occ1, u0, mo_coeff)

        def rotate_mo_freeze(newton_mf, mo_coeff1, u, log=None):
            mo = original_rotate_mo(mo_coeff1, u, log)
            if freeze_active:
                mo = np.array(mo, copy=True)
                mo[:, active_idx] = active_mo_coeff
            return mo

        soscf.gen_g_hop = MethodType(gen_g_hop_inactive, soscf)
        soscf.get_grad = MethodType(get_grad_inactive, soscf)
        soscf.update_rotate_matrix = MethodType(update_rotate_matrix_inactive, soscf)
        soscf.rotate_mo = MethodType(rotate_mo_freeze, soscf)

        options = self.merged_options(
            scf_options,
            newton_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        cycle_info = {"cycles": 0}

        def newton_callback(envs):
            if "imacro" in envs:
                cycle_info["cycles"] = max(cycle_info["cycles"], envs["imacro"] + 1)
            if callable(callback):
                callback(envs)

        with self.temporary_options(soscf, options):
            scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = newton_ah.kernel(
                soscf,
                mo_coeff,
                mo_occ,
                dm=dm0,
                conv_tol=soscf.conv_tol,
                conv_tol_grad=soscf.conv_tol_grad,
                max_cycle=soscf.max_cycle,
                dump_chk=dump_chk,
                callback=newton_callback,
                verbose=soscf.verbose,
            )

        dm = mf.make_rdm1(mo_coeff, mo_occ)
        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycle_info["cycles"],
                mode="newton_soscf",
            )
        )

    newton_soscf = newton_soscf_kernel

    def _ediis_baths(self, mo_coeff, active_idx, core_idx, active_dms,
                     max_cycle, conv_tol, conv_tol_grad, labels=None):
        """EDIIS + DIIS (lockstep) of the baths of the spin-summed active
        densities active_dms (AO), all with the reference orbitals mo_coeff.
        Returns per bath (converged, mo_coeff, cycles); the inactive columns
        hold the canonical core and virtual orbitals."""
        log = logger.new_logger(self)
        mol = self.mol
        if conv_tol_grad is None:
            # DIIS converges fast near the solution: a tighter gradient
            # than kernel() at little cost (quantities that are linear in
            # the bath orbitals, e.g. XMS energies and nuclear gradients,
            # then do not depend on the solver path); a bath that stops at
            # max_cycle below sqrt(conv_tol) is converged
            accept_grad = np.sqrt(conv_tol)
            conv_tol_grad = min(max(1e-3 * accept_grad, 1e-8), accept_grad)
        else:
            accept_grad = conv_tol_grad
        nmo = mo_coeff.shape[1]
        inact = np.setdiff1d(np.arange(nmo), active_idx)
        vir_idx = np.setdiff1d(inact, core_idx)
        orth = np.asarray(mo_coeff[:, inact], order="C")
        ncore = len(core_idx)
        u0 = np.eye(inact.size)[:, np.searchsorted(inact, core_idx)]
        if labels is None:
            labels = [str(k) for k in range(len(active_dms))]
        vj, vk = self.get_jk(mol, np.asarray(active_dms), hermi=1)
        hcore = self.get_hcore(mol)
        hprimes = [hcore + v for v in
                   np.asarray(vj - .5 * vk).reshape(np.shape(active_dms))]
        solver = _BathEDIIS(
            self, orth, ncore, hprimes, labels, log, conv_tol, conv_tol_grad,
            max_cycle, self.ediis_switch, self.diis_space,
            self.ediis_occupation, self.max_stability, self.stability_tol,
            accept_grad)
        out = []
        for b in solver.run(u0):
            st = b.st
            yo = scipy.linalg.eigh(st.foo)[1] if st.foo.size else st.foo
            yv = scipy.linalg.eigh(st.fvv)[1] if st.fvv.size else st.fvv
            mo = np.array(mo_coeff, copy=True)
            mo[:, core_idx] = lib.dot(orth, lib.dot(st.u, yo))
            mo[:, vir_idx] = lib.dot(orth, lib.dot(st.uv, yv))
            log.info("bath %s: EDIIS converged=%s (%s, %d cycles) "
                     "E_core= %.15g  |g_bath|= %4.3g  lowest Hessian "
                     "eigenvalue %s", b.label, b.conv, b.stage, b.cycles,
                     st.e, st.norm_g,
                     "-" if b.hess_min is None else "%.3g" % b.hess_min)
            out.append((bool(b.conv), mo, b.cycles))
        return out

    def _ediis_result(self, converged, mo_coeff, mo_occ, dm, cycles, mode):
        """FASSCFResult with the energy and the orbital energies of the
        Roothaan Fock matrix at mo_coeff (as the convergence check of
        kernel())."""
        mol = self.mol
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        vhf = self.get_veff(mol, dm)
        fock = np.asarray(self.get_fock(h1e, s1e, vhf, dm))
        mo_energy = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
        if np.allclose(mo_energy.imag, 0):
            mo_energy = mo_energy.real
        e_tot = self.energy_tot(dm, h1e, vhf)
        return FASSCFResult(converged, float(e_tot), mo_energy, mo_coeff,
                            mo_occ, dm, cycles, mode=mode)

    def ediis_kernel_many(
        self,
        targets,
        mo_coeff=None,
        max_cycle=None,
        conv_tol=None,
        conv_tol_grad=None,
    ):
        """EDIIS + DIIS of the baths of several active occupations (each
        target as in kernel()) from the same reference orbitals, in
        lockstep.  Returns a list of FASSCFResult (mode "ediis").  A bath
        that does not converge returns the reference orbitals if
        restore_on_failure."""
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        preps = [self.prepare_orbitals(mo_coeff, None, None, None, None, t)
                 for t in targets]
        mo_ref, _, active_idx, core_idx, _, _ = preps[0]
        cact = mo_ref[:, active_idx]
        adms = [lib.dot(cact * p[4], cact.conj().T) for p in preps]
        sols = self._ediis_baths(mo_ref, active_idx, core_idx, adms,
                                 max_cycle, conv_tol, conv_tol_grad)
        results = []
        for (conv, mo, cycles), p in zip(sols, preps):
            if not conv and self.restore_on_failure:
                mo = np.array(mo_ref, copy=True)
            mo_occ = p[5]
            dm = self.make_rdm1(mo, mo_occ)
            results.append(self._ediis_result(conv, mo, mo_occ, dm, cycles,
                                              "ediis"))
        return results

    def ediis_kernel(
        self,
        target=None,
        mo_coeff=None,
        max_cycle=None,
        conv_tol=None,
        conv_tol_grad=None,
    ):
        """Run active-frozen FASSCF with the EDIIS + DIIS solver."""
        return self.finish(self.ediis_kernel_many(
            [target], mo_coeff, max_cycle, conv_tol, conv_tol_grad)[0])

    def mixed_routine(
        self,
        target=None,
        damp=0.5,
        level_shift=0.5,
        soscf=True,
        max_cycle1=50,
        max_cycle2=100,
        dump_chk=None,
        ediis=True,
    ):
        """EDIIS + DIIS (ediis=True); if it fails, the damped and
        level-shifted kernel() and then Newton SOSCF (soscf=True)."""
        if ediis:
            result = self.ediis_kernel(target=target, max_cycle=max_cycle1)
            if result.converged:
                return result
            logger.info(self, "FASSCF: EDIIS + DIIS not converged; "
                        "damped iterations follow")
        options = {"damp": damp, "level_shift": level_shift}
        result = self.kernel(
            target=target,
            max_cycle=max_cycle1,
            dump_chk=dump_chk,
            scf_options=options,
        )

        if soscf and not self.converged:
            return self.newton_soscf_kernel(
                target=target,
                max_cycle=max_cycle2,
                dump_chk=dump_chk,
            )
        return result


    def prepare_orbitals(
        self,
        mo_coeff,
        mo_energy,
        active_orbitals,
        core_orbitals,
        active_occ,
        target=None,
    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            mo_coeff = getattr(self, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("mo_coeff is required.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("FASSCF expects a 2D MO coefficient array.")

        nmo = mo_coeff.shape[1]
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_energy is None:
            mo_energy = getattr(self, "mo_energy", None)
        if mo_energy is None:
            mo_energy = np.zeros(nmo)
        else:
            mo_energy = np.asarray(mo_energy)

        active_orbitals = (
            self.active_orbitals if active_orbitals is None else active_orbitals
        )
        core_orbitals = self.core_orbitals if core_orbitals is None else core_orbitals
        active_idx = self.validate_indices(active_orbitals, nmo, "active_orbitals")
        core_idx = self.validate_indices(core_orbitals, nmo, "core_orbitals")

        overlap = np.intersect1d(active_idx, core_idx)
        if overlap.size:
            raise ValueError(f"active_orbitals and core_orbitals overlap: {overlap.tolist()}")

        if active_occ is None:
            active_occ = self.active_occ_from_target(target)
        if active_occ is None:
            raise ValueError("active_occ is required.")
        active_occ = np.asarray(active_occ, dtype=float)
        if active_occ.size != active_idx.size:
            raise ValueError(
                f"active_occ has length {active_occ.size}, expected {active_idx.size}."
            )

        mo_occ = np.zeros(nmo)
        mo_occ[core_idx] = 2.0
        mo_occ[active_idx] = active_occ
        return mo_coeff, mo_energy, active_idx, core_idx, active_occ, mo_occ

    def validate_indices(self, indices, nmo, name):
        if indices is None:
            raise ValueError(f"{name} is required.")
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size != np.unique(idx).size:
            raise ValueError(f"{name} contains duplicate indices.")
        if idx.size and (idx.min() < 0 or idx.max() >= nmo):
            raise ValueError(f"{name} contains indices outside 0..{nmo - 1}.")
        return np.sort(idx)

    def merged_options(self, scf_options, direct_options, **runtime_options):
        options = dict(self.scf_options)
        options.update(scf_options or {})
        options.update(direct_options)
        for key, value in runtime_options.items():
            if value is not None:
                options[key] = value
        return options

    @contextmanager
    def temporary_options(self, obj, options):
        old_values = {}
        for key, value in options.items():
            old_values[key] = getattr(obj, key, MISSING)
            setattr(obj, key, value)
        try:
            yield
        finally:
            for key, old_value in old_values.items():
                if old_value is MISSING:
                    try:
                        delattr(obj, key)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, key, old_value)

    def make_diis(self, mf, h1e, s1e, vhf, dm):
        if isinstance(getattr(mf, "diis", None), lib.diis.DIIS):
            return mf.diis
        if not getattr(mf, "diis", False):
            return None

        diis_class = getattr(mf, "DIIS", None)
        if diis_class is None:
            return None
        mf_diis = diis_class(mf, getattr(mf, "diis_file", None))
        if hasattr(mf_diis, "space"):
            mf_diis.space = getattr(mf, "diis_space", mf_diis.space)
        if hasattr(mf_diis, "rollback"):
            mf_diis.rollback = getattr(mf, "diis_space_rollback", mf_diis.rollback)

        try:
            fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
            if fock.ndim != 2:
                raise ValueError(FOCK_ERROR)
            _, mf_diis.Corth = mf.eig(fock, s1e)
        except Exception as exc:
            logger.debug(mf, "Unable to initialize DIIS Corth: %s", exc)
        return mf_diis

    def log_overlap_condition(self, mf, s1e, conv_tol):
        cond = lib.cond(s1e)
        logger.debug(mf, "cond(S) = %s", cond)
        if np.max(cond) * 1e-17 > conv_tol:
            logger.warn(
                mf,
                "Singularity detected in overlap matrix (condition number = %4.3g). "
                "SCF may be inaccurate and hard to converge.",
                np.max(cond),
            )

    def call_hook(self, obj, name, arg):
        method = getattr(obj, name, None)
        if callable(method):
            return method(arg)
        return None

    def finish(self, result):
        self.converged = result.converged
        self.e_tot = result.e_tot
        self.mo_energy = result.mo_energy
        self.mo_coeff = result.mo_coeff
        self.mo_occ = result.mo_occ
        self.dm = result.dm
        self.cycles = result.cycles
        self.last_result = result
        return result



class GroupAverageFASSCF(FASSCF):
    """FASSCF subclass for group-averaged active-space densities."""

    def kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        conv_check=None,
        scf_options=None,
        **scf_kwargs,
    ):
        """Run group-average FASSCF using this object as the SCF object."""
        if "init_dm" in scf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        if not isinstance(mf, rohf.ROHF):
            raise TypeError("FASSCF group-average mode requires a PySCF ROHF-like object.")

        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        conv_check = self.conv_check if conv_check is None else conv_check
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci

        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average FASSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average FASSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average FASSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        initial_mo_coeff = np.array(mo_coeff, copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average FASSCF.")

        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_mo_coeff = mo_coeff[:, active_idx]
        as_dm_a = (as_mo_coeff * avg_occa).dot(as_mo_coeff.conj().T)
        as_dm_b = (as_mo_coeff * avg_occb).dot(as_mo_coeff.conj().T)
        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        options = self.merged_options(
            scf_options,
            scf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(mf, options):
            if dm0 is None:
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
            else:
                dm = dm0
            h1e = mf.get_hcore(mol)
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            s1e = mf.get_ovlp(mol)
            self.log_overlap_condition(mf, s1e, conv_tol)
            mf_diis = self.make_diis(mf, h1e, s1e, vhf, dm)
            cput1 = logger.timer(mf, "initialize scf", *cput0)
            fock_last = None
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = np.asarray(
                    mf.get_fock(
                        h1e, s1e, vhf, dm,
                        cycle=cycle, diis=mf_diis, fock_last=fock_last,
                    )
                )
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                fock_last = fock

                fock_plain = mf.get_fock(h1e, s1e, vhf, dm)
                if np.asarray(fock_plain).ndim != 2:
                    raise ValueError(FOCK_ERROR)
                g_inact = _inactive_orbital_gradient(
                    mf, mo_coeff, mo_occ, fock_plain, active_idx)
                norm_gorb = np.linalg.norm(g_inact)
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                logger.info(
                    mf,
                    "cycle= %d E= %.15g  delta_E= %4.3g "
                    "|g_bath|= %4.3g |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )
                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                elif (abs(e_tot - last_hf_e) < conv_tol and
                      norm_gorb < conv_tol_grad):
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                cput1 = logger.timer(mf, f"cycle= {cycles}", *cput1)
                if scf_conv:
                    break

            if scf_conv and conv_check:
                dm_last = dm
                last_hf_e = e_tot
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                fock_plain = mf.get_fock(h1e, s1e, vhf, dm)
                if np.asarray(fock_plain).ndim != 2:
                    raise ValueError(FOCK_ERROR)
                g_inact = _inactive_orbital_gradient(
                    mf, mo_coeff, mo_occ, fock_plain, active_idx)
                norm_gorb = np.linalg.norm(g_inact)
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                loose_e_tol = conv_tol * 10
                loose_g_tol = conv_tol_grad * 3
                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                else:
                    scf_conv = (abs(e_tot - last_hf_e) < loose_e_tol or
                                norm_gorb < loose_g_tol)
                logger.info(
                    mf,
                    "Extra cycle  E= %.15g  delta_E= %4.3g "
                    "|g_bath|= %4.3g |ddm|= %4.3g",
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )
                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())

            logger.timer(mf, "scf_cycle", *cput0)

        if not scf_conv and self.restore_on_failure:
            mo_coeff = initial_mo_coeff

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="group_average",
            )
        )

    group_average = kernel

    def ediis_kernel_many(
        self,
        target_groups,
        group_info_list,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        max_cycle=None,
        conv_tol=None,
        conv_tol_grad=None,
    ):
        """EDIIS + DIIS of the group-average baths of several groups in
        lockstep.  Returns a list of FASSCFResult (mode
        "group_average_ediis").  A bath that does not converge returns the
        reference orbitals if restore_on_failure."""
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        gbci = self.gbci
        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)
        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = int(self.ncas if ncas is None else ncas)
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = int(self.ncore if ncore is None else ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        cact = mo_coeff[:, active_idx]
        occs = [_group_average_occ(ncas, nelecas, group_info_list, g)
                for g in target_groups]
        as_dms = [((cact * oa).dot(cact.conj().T), (cact * ob).dot(cact.conj().T))
                  for oa, ob in occs]
        sols = self._ediis_baths(
            mo_coeff, active_idx, np.arange(ncore), [a + b for a, b in as_dms],
            max_cycle, conv_tol, conv_tol_grad,
            labels=[str(g) for g in target_groups])
        results = []
        for (conv, mo, cycles), (oa, ob), (dma, dmb) in zip(sols, occs, as_dms):
            if not conv and self.restore_on_failure:
                mo = np.array(mo_coeff, copy=True)
            mo_occ = np.zeros(nmo)
            mo_occ[:ncore] = 2.0
            mo_occ[active_idx] = oa + ob
            core = mo[:, :ncore].dot(mo[:, :ncore].conj().T)
            dm = np.asarray((core + dma, core + dmb))
            results.append(self._ediis_result(conv, mo, mo_occ, dm, cycles,
                                              "group_average_ediis"))
        return results

    def ediis_kernel(
        self,
        target_group,
        group_info_list,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        max_cycle=None,
        conv_tol=None,
        conv_tol_grad=None,
    ):
        """Run group-average FASSCF with the EDIIS + DIIS solver."""
        return self.finish(self.ediis_kernel_many(
            [target_group], group_info_list, mo_coeff, ncas, nelecas, ncore,
            max_cycle, conv_tol, conv_tol_grad)[0])

    def mixed_routine(
        self,
        target_group,
        group_info_list,
        damp=0.5,
        level_shift=0.5,
        soscf=True,
        max_cycle1=100,
        max_cycle2=300,
        dump_chk=None,
        ediis=True,
    ):
        """EDIIS + DIIS (ediis=True); if it fails, the damped and
        level-shifted kernel() and then Newton SOSCF (soscf=True)."""
        if ediis:
            result = self.ediis_kernel(target_group, group_info_list,
                                       max_cycle=max_cycle1)
            if result.converged:
                return result
            logger.info(self, "FASSCF: EDIIS + DIIS not converged; "
                        "damped iterations follow")
        options = {"damp": damp, "level_shift": level_shift}
        result = self.kernel(
            target_group=target_group,
            group_info_list=group_info_list,
            max_cycle=max_cycle1,
            dump_chk=dump_chk,
            scf_options=options,
        )

        if soscf and not self.converged:
            return self.newton_soscf_kernel(
                target_group=target_group,
                group_info_list=group_info_list,
                max_cycle=max_cycle2,
                dump_chk=dump_chk,
            )
        return result

    def soscf_kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        freeze_active=True,
        canonicalization=False,
        scf_options=None,
        **soscf_kwargs,
    ):
        """Run inactive-only SOSCF with group-averaged active density."""
        if "init_dm" in soscf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        if not isinstance(mf, rohf.ROHF):
            raise TypeError("FASSCF group-average SOSCF requires a PySCF ROHF-like object.")

        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci

        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average SOSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average SOSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average SOSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average SOSCF.")

        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_dm_a = (active_mo_coeff * avg_occa).dot(active_mo_coeff.conj().T)
        as_dm_b = (active_mo_coeff * avg_occb).dot(active_mo_coeff.conj().T)

        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        soscf = mf.newton()
        options = self.merged_options(
            scf_options,
            soscf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(soscf, options):
            h1e = mf.get_hcore(mol)
            s1e = mf.get_ovlp(mol)
            if dm0 is None:
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
            else:
                dm = dm0
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "group-average SOSCF init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                g_full, h_op_full, h_diag_full = soscf.gen_g_hop(
                    mo_coeff, mo_occ, fock
                )

                active_mask = np.zeros(nmo, dtype=bool)
                active_mask[active_idx] = True
                occidxa = mo_occ > 0
                occidxb = mo_occ == 2
                viridxa = ~occidxa
                viridxb = ~occidxb
                uniq_var_a = viridxa[:, None] & occidxa[None, :]
                uniq_var_b = viridxb[:, None] & occidxb[None, :]
                uniq_ab = uniq_var_a | uniq_var_b
                if freeze_active:
                    inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
                else:
                    inactive_rotation = np.ones((nmo, nmo), dtype=bool)
                inactive_var = inactive_rotation[uniq_ab]

                g_full = np.asarray(g_full)
                if inactive_var.size != g_full.size:
                    raise RuntimeError(
                        "Unable to map group-average SOSCF variables to inactive-only space."
                    )
                g_orb = g_full[inactive_var]
                h_diag = np.asarray(h_diag_full)[inactive_var]
                norm_gorb = np.linalg.norm(g_orb)

                if g_orb.size == 0:
                    scf_conv = True
                    break

                def h_op(x):
                    x_full = np.zeros_like(g_full)
                    x_full[inactive_var] = x
                    return np.asarray(h_op_full(x_full))[inactive_var]

                def g_op():
                    return g_orb

                def precond(x, e):
                    hdiagd = h_diag - (e - soscf.ah_level_shift)
                    hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                    return x / hdiagd

                dx = np.zeros_like(g_orb)
                ah_conv_tol = min(norm_gorb**2, soscf.ah_conv_tol)
                ah_start_tol = min(norm_gorb * 5, soscf.ah_start_tol)
                for ah_end, ihop, w, dxi, hdxi, residual, seig in ciah.davidson_cc(
                    h_op,
                    g_op,
                    precond,
                    g_orb,
                    tol=ah_conv_tol,
                    max_cycle=soscf.ah_max_cycle,
                    lindep=soscf.ah_lindep,
                    verbose=logger.new_logger(soscf, soscf.verbose),
                ):
                    if (
                        ah_end
                        or ihop == soscf.ah_max_cycle
                        or (
                            np.linalg.norm(residual) < ah_start_tol
                            and ihop >= soscf.ah_start_cycle
                        )
                        or seig < soscf.ah_lindep
                    ):
                        dx = np.array(dxi, copy=True)
                        dxmax = np.max(abs(dx)) if dx.size else 0
                        if dxmax > soscf.max_stepsize:
                            dx *= soscf.max_stepsize / dxmax
                        break

                dx_full = np.zeros_like(g_full)
                dx_full[inactive_var] = dx
                u = soscf.update_rotate_matrix(dx_full, mo_occ, mo_coeff=mo_coeff)
                mo_coeff = soscf.rotate_mo(mo_coeff, u)
                if freeze_active:
                    mo_coeff[:, active_idx] = active_mo_coeff

                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                fock_mo = mo_coeff.conj().T @ np.asarray(fock) @ mo_coeff
                mo_energy = np.diag(fock_mo)
                if np.allclose(mo_energy.imag, 0):
                    mo_energy = mo_energy.real
                e_tot = mf.energy_tot(dm, h1e, vhf)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                logger.info(
                    mf,
                    "group-average SOSCF cycle= %d E= %.15g  delta_E= %4.3g  |g_bath|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                if scf_conv:
                    break

            logger.timer(mf, "group-average inactive-only SOSCF", *cput0)

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="group_average_soscf",
            )
        )

    def newton_soscf_kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        freeze_active=True,
        canonicalization=False,
        scf_options=None,
        **newton_kwargs,
    ):
        """Run PySCF Newton SOSCF with group-averaged active density."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci
        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average Newton SOSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average Newton SOSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average Newton SOSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average Newton SOSCF.")
        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_dm_a = (active_mo_coeff * avg_occa).dot(active_mo_coeff.conj().T)
        as_dm_b = (active_mo_coeff * avg_occb).dot(active_mo_coeff.conj().T)

        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        soscf = mf.newton()
        original_gen_g_hop = soscf.gen_g_hop
        original_get_grad = soscf.get_grad
        original_update_rotate_matrix = soscf.update_rotate_matrix
        original_rotate_mo = soscf.rotate_mo

        def make_group_average_dm(newton_mf, mo_coeff1, mo_occ1=None):
            core_mo_coeff = mo_coeff1[:, :ncore]
            dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
            return np.asarray(
                (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
            )

        def gen_g_hop_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None, *args, **kwargs):
            g_full, h_op_full, h_diag_full = original_gen_g_hop(
                mo_coeff1, mo_occ1, fock_ao, *args, **kwargs
            )
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]

            g_full = np.asarray(g_full)
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map group-average Newton SOSCF variables to inactive-only space."
                )
            g_orb = g_full[inactive_var]
            h_diag = np.asarray(h_diag_full)[inactive_var]

            def h_op(x):
                x_full = np.zeros_like(g_full)
                x_full[inactive_var] = x
                return np.asarray(h_op_full(x_full))[inactive_var]

            return g_orb, h_op, h_diag

        def get_grad_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None):
            g_full = np.asarray(original_get_grad(mo_coeff1, mo_occ1, fock_ao))
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map group-average Newton SOSCF gradient to inactive-only space."
                )
            return g_full[inactive_var]

        def update_rotate_matrix_inactive(
            newton_mf, dx, mo_occ1, u0=1, mo_coeff=None
        ):
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if np.asarray(dx).size == int(np.count_nonzero(inactive_var)):
                dx_full = np.zeros(inactive_var.size, dtype=np.asarray(dx).dtype)
                dx_full[inactive_var] = dx
            else:
                dx_full = dx
            return original_update_rotate_matrix(dx_full, mo_occ1, u0, mo_coeff)

        def rotate_mo_freeze(newton_mf, mo_coeff1, u, log=None):
            mo = original_rotate_mo(mo_coeff1, u, log)
            if freeze_active:
                mo = np.array(mo, copy=True)
                mo[:, active_idx] = active_mo_coeff
            return mo

        soscf.make_rdm1 = MethodType(make_group_average_dm, soscf)
        soscf.gen_g_hop = MethodType(gen_g_hop_inactive, soscf)
        soscf.get_grad = MethodType(get_grad_inactive, soscf)
        soscf.update_rotate_matrix = MethodType(update_rotate_matrix_inactive, soscf)
        soscf.rotate_mo = MethodType(rotate_mo_freeze, soscf)

        options = self.merged_options(
            scf_options,
            newton_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        cycle_info = {"cycles": 0}

        def newton_callback(envs):
            if "imacro" in envs:
                cycle_info["cycles"] = max(cycle_info["cycles"], envs["imacro"] + 1)
            if callable(callback):
                callback(envs)

        with self.temporary_options(soscf, options):
            scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = newton_ah.kernel(
                soscf,
                mo_coeff,
                mo_occ,
                dm=dm0,
                conv_tol=soscf.conv_tol,
                conv_tol_grad=soscf.conv_tol_grad,
                max_cycle=soscf.max_cycle,
                dump_chk=dump_chk,
                callback=newton_callback,
                verbose=soscf.verbose,
            )

        dm = make_group_average_dm(soscf, mo_coeff, mo_occ)
        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycle_info["cycles"],
                mode="group_average_newton_soscf",
            )
        )

    group_average_newton_soscf_kernel = newton_soscf_kernel
    group_average_newton_soscf = newton_soscf_kernel
    group_average_soscf_kernel = soscf_kernel
    group_average_soscf = soscf_kernel


def fasscf_kernel(gbci, target, **kwargs):
    """Function-style compatibility wrapper for ordinary FASSCF."""
    driver = FASSCF(gbci)
    return driver.kernel(target=target, **kwargs).as_tuple()


def state_average_fasscf_kernel(fasscf, target_group, group_info_list, **kwargs):
    """Function-style compatibility wrapper for group-average FASSCF."""
    if not isinstance(fasscf, GroupAverageFASSCF):
        raise TypeError(
            "state_average_fasscf_kernel expects a GroupAverageFASSCF object first."
        )
    return fasscf.group_average_kernel(
        target_group=target_group,
        group_info_list=group_info_list,
        **kwargs,
    ).as_tuple()


def state_average_soscf_kernel(fasscf, target_group, group_info_list, **kwargs):
    """Function-style compatibility wrapper for group-average SOSCF."""
    if not isinstance(fasscf, GroupAverageFASSCF):
        raise TypeError(
            "state_average_soscf_kernel expects a GroupAverageFASSCF object first."
        )
    return fasscf.group_average_soscf_kernel(
        target_group=target_group,
        group_info_list=group_info_list,
        **kwargs,
    ).as_tuple()
