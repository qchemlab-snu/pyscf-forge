#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0
#
# Author: Byungjoo Kim <creeperkim28@snu.ac.kr>
#         Minseok Oh <msjeff2001@snu.ac.kr>
#

'''
Linearized GB-PDFT (L-GBPDFT): L-PDFT (Hennefarth et al., JCTC 2023) on GBCI
states, with the exact one-step GBCI on-top pair density.

E_ot is expanded to first order about a reference (rho_0, Pi_0) and the
classical Coulomb energy likewise; the result is a Hermitian operator whose
matrix in the GBCI root basis is

    L_II = h H_II + (1 - h) [E_nn + tr((h_core + J_0) D_I) - tr(J_0 D_0) / 2]
           + E_ot[rho_0, Pi_0] + int v_rho (rho_I - rho_0) + v_Pi (Pi_I - Pi_0)
    L_IJ = h H_IJ + (1 - h) tr((h_core + J_0) D_IJ)
           + int v_rho rho_IJ + v_Pi Pi_IJ

with h the hybrid fraction of the on-top functional (tPBE0: 1/4) and H the
full-Coulomb GBCI Hamiltonian.  D, rho and Pi of a GBCI vector are quadratic
in its coefficients, so the transition quantities follow by polarization,
X_IJ = (X[c_I + c_J] - X[c_I] - X[c_J]) / 2, and need nothing beyond the
state routines.

method="LIN": one reference, the weighted average of the model states.
method="PLIN": L_II at the state's own density (L_II = GB-PDFT energy) and
L_IJ at the average of states I and J (pairwise L-PDFT).

ontop="cas_cumulant": approximate Pi, Pi = rho_a rho_b + Pi^CAS
- rho^CAS_a rho^CAS_b (Pi^CAS, rho^CAS: the CI vector read as a CASCI
vector).  The products are not quadratic in the coefficients, so each is
linearized about the reference, rho_a rho_b -> rho_a0 rho_b + rho_a rho_b0
- rho_a0 rho_b0; the operator is then linear again and PLIN's diagonal is
still the state-specific energy.
'''

import numpy as np
from scipy import linalg

from pyscf.gbci import gbsrpdft, otpd
from pyscf.gbci import rdm as gbci_rdm


def _hybrid(ot, spin):
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError("hybrid exchange and correlation differ")
    return float(hyb[0])


class _Grid(object):
    '''rho (with the derivatives the functional needs) and Pi of a set of
    GBCI vectors, block by block on the functional's grid.'''

    def __init__(self, mc, ot, mo, otpd_data, vecs, max_memory,
                 ontop="exact"):
        self.mc, self.ot, self.mo = mc, ot, mo
        self.data = otpd_data
        self.ontop = ontop
        self.vecs = [np.asarray(v) for v in vecs]
        self.dm1s = [np.asarray(gbci_rdm.make_rdm1s_precomputed(
            v, otpd_data, mo_coeff=mo)) for v in self.vecs]
        if ontop == "exact":
            self.caches = [otpd.make_root_intermediates(v, otpd_data)
                           for v in self.vecs]
        elif ontop == "cas_cumulant":
            from pyscf.fci import direct_spin1
            if ot.Pi_deriv != 0:
                raise NotImplementedError("cas_cumulant with Pi derivatives")
            ncas, nelecas = mc.ncas, mc.nelecas
            self.mo_cas = mo[:, mc.ncore:mc.ncore + ncas]
            self.cas_dm1s = [np.einsum("pu,suv,qv->spq", self.mo_cas,
                                       direct_spin1.make_rdm1s(v, ncas, nelecas),
                                       self.mo_cas) for v in self.vecs]
            self.cas_dm2 = [direct_spin1.make_rdm12(v, ncas, nelecas)[1]
                            for v in self.vecs]
        else:
            raise ValueError("ontop must be 'exact' or 'cas_cumulant'")
        self.max_memory = max_memory

    def blocks(self, idx=None):
        '''rho and Pi of the vectors ``idx`` (default all) per block.'''
        ot = self.ot
        ni = ot._numint
        nao = self.mo.shape[0]
        if idx is None:
            idx = range(len(self.vecs))
        idx = list(idx)
        make_rho = [[ni._gen_rho_evaluator(ot.mol, self.dm1s[k][s], 1)
                     for s in range(2)] for k in idx]
        for ao, mask, weight, _ in ni.block_loop(
                ot.mol, ot.grids, nao, ot.dens_deriv, self.max_memory):
            rho = [np.asarray([m[s][0](0, ao, mask, ot.xctype)
                               for s in range(2)]) for m in make_rho]
            rho = [r[:, None, :] if r.ndim == 2 else r for r in rho]
            if self.ontop == "exact":
                pi = [otpd.get_ontop_pair_density(
                    ot, ao, self.vecs[k], self.data, self.mo, ot.Pi_deriv,
                    mask, ot_root_cache=self.caches[k]) for k in idx]
                pi = [p[None, :] if p.ndim == 1 else p for p in pi]
            else:
                # (rho^CAS_a, rho^CAS_b, Pi^CAS) per vector, values only
                from pyscf.mcpdft import otpd as mcotpd
                ao0 = ao[0] if ao.ndim == 3 else ao
                ni0 = ot._numint
                pi = []
                for k in idx:
                    rc = np.asarray([ni0.eval_rho(ot.mol, ao0, d, mask, "LDA",
                                                  hermi=1)
                                     for d in self.cas_dm1s[k]])
                    p2 = mcotpd.get_ontop_pair_density(
                        ot, np.zeros((2, 1, rc.shape[-1])), ao,
                        self.cas_dm2[k], self.mo_cas, 0, mask)
                    pi.append((rc, np.asarray(p2).ravel()))
            yield weight, rho, pi
            del ao


def _ot_terms(ot, grid, w0, need=None):
    '''For reference weights w0 over grid.vecs (a density that is the
    weighted sum of those vectors' rho and Pi): E_ot[rho_0, Pi_0] and
    int v . X for X = rho_k, Pi_k of the vectors in ``need`` (default all).
    Returns (E_ot0, lin, lin_ref): lin[k] = int v_rho rho_k + v_Pi Pi_k
    (nan outside ``need``), lin_ref = int v . (rho_0, Pi_0).'''
    ref = [k for k, w in enumerate(w0) if w]
    if need is None:
        need = range(len(grid.vecs))
    idx = sorted(set(ref) | set(need))
    pos = {k: p for p, k in enumerate(idx)}
    e0 = 0.0
    lin = np.full(len(grid.vecs), np.nan)
    lin[list(need)] = 0.0
    lin_ref = 0.0
    for weight, rho_l, pi_l in grid.blocks(idx):
        rho = {k: rho_l[pos[k]] for k in idx}
        pi = {k: pi_l[pos[k]] for k in idx}
        rho0 = sum(w0[k] * rho[k] for k in ref)
        if grid.ontop == "cas_cumulant":
            # Pi = rho_a rho_b + Pi^CAS - rho^CAS_a rho^CAS_b, each product
            # linearized about the reference: pi[k] is the part linear in
            # vector k, const what the reference adds once
            ra0, rb0 = rho0[0, 0], rho0[1, 0]
            ca0 = sum(w0[k] * pi[k][0][0] for k in ref)
            cb0 = sum(w0[k] * pi[k][0][1] for k in ref)
            pi = {k: (ra0 * rho[k][1, 0] + rho[k][0, 0] * rb0 + pi[k][1]
                      - ca0 * pi[k][0][1] - pi[k][0][0] * cb0)[None, :]
                  for k in idx}
            const = ca0 * cb0 - ra0 * rb0
        else:
            const = 0.0
        pi_lin0 = sum(w0[k] * pi[k] for k in ref)
        pi0 = pi_lin0 + const
        eot, vot, _ = ot.eval_ot(rho0, pi0, dderiv=1, weights=weight)
        # translated functionals: v_rho is the derivative with respect to
        # the total density and its gradient (rows of vrho), v_Pi with
        # respect to Pi (and its derivatives, if any)
        vrho, vpi = (np.asarray(v) for v in vot)
        nr, npi = vrho.shape[0], vpi.shape[0]
        e0 += eot.dot(weight)

        def dot(r, p):
            return (np.einsum("dg,dg,g->", vrho, r.sum(0)[:nr], weight)
                    + np.einsum("dg,dg,g->", vpi, p[:npi], weight))
        for k in need:
            lin[k] += dot(rho[k], pi[k])
        lin_ref += dot(rho0, pi_lin0)
    return e0, lin, lin_ref


def kernel(mc, ci_list, mo_coeff=None, ot=None, intermediates=None,
           weights=None, method="LIN", ontop="exact"):
    '''L-GBPDFT on the GBCI states ``ci_list`` (orthonormal roots of the
    full-Coulomb GBCI).  Returns e_states, si, heff (the L matrix), ci.'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ot is None:
        ot = mc.otfnal
    inter = gbsrpdft._intermediates(mc, mo_coeff, intermediates)
    ci = [np.asarray(c) for c in ci_list]
    n = len(ci)
    if weights is None:
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    ot.reset(mol=mc.mol)
    h = _hybrid(ot, abs(mc.nelecas[0] - mc.nelecas[1]))
    otpd_data = mc.make_otpd_intermediates(mo_coeff, intermediates=inter)

    # states first, then the pair sums c_I + c_J for the transition terms
    pairs = [(i, j) for i in range(n) for j in range(i)]
    vecs = ci + [ci[i] + ci[j] for i, j in pairs]
    grid = _Grid(mc, ot, mo_coeff, otpd_data, vecs, mc.max_memory, ontop)
    dm = [d[0] + d[1] for d in grid.dm1s]

    hmat = gbsrpdft.hamiltonian_matrix(
        mc, gbsrpdft.full_hamiltonian(mc, mo_coeff, inter), inter, ci)
    hcore = mc._scf.get_hcore()
    enuc = mc._scf.energy_nuc()

    def transition(x, i, j, k):
        # x[k] = X[c_i + c_j]; X_ij = (X[c_i + c_j] - X[c_i] - X[c_j]) / 2
        return 0.5 * (x[k] - x[i] - x[j])

    def l_matrix(w0, rows):
        '''L elements (i, j) in ``rows`` for the reference sum_k w0_k X_k.'''
        dm0 = sum(w * d for w, d in zip(w0, dm) if w)
        j0 = mc._scf.get_j(dm=dm0)
        need = set()
        for i, j in rows:
            need |= {i, j}
            if i != j:
                need.add(n + pairs.index((max(i, j), min(i, j))))
        e_ot0, lin, lin_ref = _ot_terms(ot, grid, w0, sorted(need))
        hj = hcore + j0
        tr = np.array([np.tensordot(hj, d) for d in dm])
        out = {}
        for i, j in rows:
            if i == j:
                out[i, j] = (h * hmat[i, i]
                             + (1 - h) * (enuc + tr[i]
                                          - 0.5 * np.tensordot(j0, dm0))
                             + e_ot0 + lin[i] - lin_ref)
            else:
                k = n + pairs.index((max(i, j), min(i, j)))
                out[i, j] = (h * hmat[i, j]
                             + (1 - h) * transition(tr, i, j, k)
                             + transition(lin, i, j, k))
        return out

    L = np.zeros((n, n))
    if method.upper() == "LIN":
        w0 = np.concatenate([weights, np.zeros(len(pairs))])
        el = l_matrix(w0, [(i, j) for i in range(n) for j in range(i + 1)])
        for (i, j), v in el.items():
            L[i, j] = L[j, i] = v
    elif method.upper() == "PLIN":
        for i in range(n):
            w0 = np.zeros(len(vecs))
            w0[i] = 1.0
            L[i, i] = l_matrix(w0, [(i, i)])[i, i]
        for i, j in pairs:
            w0 = np.zeros(len(vecs))
            w0[i] = w0[j] = 0.5
            L[i, j] = L[j, i] = l_matrix(w0, [(i, j)])[i, j]
    else:
        raise ValueError("method must be 'LIN' or 'PLIN'")
    e, si = linalg.eigh(L)
    return {"e_states": e, "si": si, "heff": L,
            "ci": list(np.tensordot(si.T, np.asarray(ci), axes=1)),
            "method": method.upper(), "ontop": ontop}
